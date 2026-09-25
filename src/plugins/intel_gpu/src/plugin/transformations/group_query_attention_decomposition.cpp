// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "group_query_attention_decomposition.hpp"

#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::intel_gpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;

namespace {

bool is_supported_compressed_kv_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 || type == ov::element::i4 ||
           type == ov::element::u4;
}

std::vector<uint64_t> compute_kv_group_sizes(const ov::PartialShape& data_shape,
                                             const ov::PartialShape& scale_shape) {
    if (data_shape.rank().is_dynamic()) {
        return {};
    }

    const size_t rank = data_shape.rank().get_length();
    std::vector<uint64_t> group_sizes(rank, 1);
    if (scale_shape.rank().is_static() && static_cast<size_t>(scale_shape.rank().get_length()) == rank) {
        for (size_t i = 0; i < rank; ++i) {
            const bool scale_is_one = scale_shape[i].is_static() && scale_shape[i].get_length() == 1;
            const bool data_is_one = data_shape[i].is_static() && data_shape[i].get_length() == 1;
            if (scale_is_one && !data_is_one) {
                group_sizes[i] = std::numeric_limits<uint64_t>::max();
            }
        }
    } else if (rank > 0) {
        group_sizes[rank - 1] = std::numeric_limits<uint64_t>::max();
    }
    return group_sizes;
}

}  // namespace

GroupQueryAttentionDecomposition::KVCacheOutputs GroupQueryAttentionDecomposition::construct_kvcache(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
    const ov::Output<ov::Node>& past_key,
    const ov::Output<ov::Node>& past_value,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& seqlens_1d,
    const ov::Output<ov::Node>& past_seqlen,
    const ov::Output<ov::Node>& /*current_seqlen_scalar*/) {
    m_use_compressed_sdpa = false;
    const auto window_size = node->get_sliding_window_cache() ? node->get_local_window_size() : 0;
    const auto key_cache = register_new_node<op::StatelessKV>(past_key, key, seqlens_1d, 2, true, window_size);
    const auto value_transpose_order =
        register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2}));
    const auto find_value_transpose = [](const ov::Output<ov::Node>& output) {
        const auto transpose = ov::as_type_ptr<v1::Transpose>(output.get_node_shared_ptr());
        if (!transpose) {
            return std::shared_ptr<v1::Transpose>{};
        }
        const auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (!order || order->cast_vector<int64_t>() != std::vector<int64_t>{0, 1, 3, 2}) {
            return std::shared_ptr<v1::Transpose>{};
        }
        return transpose;
    };

    const auto input_past_value = node->input_value(4);
    const auto past_value_transpose = find_value_transpose(input_past_value);
    std::shared_ptr<v1::Transpose> present_value_transpose;
    if (node->output(2).get_target_inputs().size() == 1) {
        const auto& target = *node->output(2).get_target_inputs().begin();
        present_value_transpose = find_value_transpose(target.get_node()->output(target.get_index()));
    }

    m_transpose_v = past_value_transpose && present_value_transpose;
    int64_t value_seq_axis = 2;
    std::shared_ptr<op::StatelessKV> value_cache;
    if (m_transpose_v) {
        // Before: past_value -> Transpose -> GQA cache -> Transpose -> present_value.
        // After:  past_value -------------------------> StatelessKV(axis=3) -> present_value.
        //         current_value -> Transpose ----------^      (the output Transpose is bypassed)
        value_seq_axis = 3;
        const auto transposed_value = register_new_node<v1::Transpose>(value, value_transpose_order);
        value_cache = register_new_node<op::StatelessKV>(past_value_transpose->input_value(0),
                                                         transposed_value,
                                                         seqlens_1d,
                                                         value_seq_axis,
                                                         true,
                                                         window_size);
        present_value_transpose->output(0).replace(value_cache->output(0));
    } else {
        value_cache =
            register_new_node<op::StatelessKV>(past_value, value, seqlens_1d, value_seq_axis, true, window_size);
    }
    KVCacheOutputs outputs;
    outputs.present_key = key_cache->output(0);
    outputs.present_value = value_cache->output(0);
    outputs.sdpa_key = key_cache->output(1);
    outputs.sdpa_value = value_cache->output(1);
    if (node->is_kv_quantized() && node->get_kv_cache_bit_width() == 8 &&
        past_key.get_element_type() == past_value.get_element_type() &&
        is_supported_compressed_kv_type(past_key.get_element_type())) {
        using GQAInputs = ov::op::internal::GroupQueryAttentionInputs;
        m_compressed_key = outputs.sdpa_key;
        m_compressed_value = outputs.sdpa_value;
        m_key_scale = make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::K_SCALE)),
                                    node->get_kv_num_heads(),
                                    node->get_k_quant_type());
        m_value_scale = make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::V_SCALE)),
                                      node->get_kv_num_heads(),
                                      node->get_v_quant_type());
        if (m_key_scale.get_element_type() != ov::element::f16) {
            m_key_scale = register_new_node<v0::Convert>(m_key_scale, ov::element::f16);
        }
        if (m_value_scale.get_element_type() != ov::element::f16) {
            m_value_scale = register_new_node<v0::Convert>(m_value_scale, ov::element::f16);
        }
        m_quantization_attrs.quantization_type =
            ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        m_quantization_attrs.output_storage_type =
            ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        m_quantization_attrs.quantization_dt = past_key.get_element_type();
        m_quantization_attrs.scale_dt = ov::element::f16;
        m_quantization_attrs.group_sizes =
            compute_kv_group_sizes(m_compressed_key.get_partial_shape(), m_key_scale.get_partial_shape());
        m_quantization_attrs.scales_zp_output_order = {0, 1, 2, 3};
        m_use_compressed_sdpa = true;
    }

    outputs.mask_past_seqlen = past_seqlen;
    outputs.bias_col_offset = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

    if (node->get_sliding_window_cache()) {
        const auto zero_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        const auto past_scalar = register_new_node<v0::Squeeze>(past_seqlen);
        const auto window_minus_one = register_new_node(
            v0::Constant::create(ov::element::i64, ov::Shape{}, {node->get_local_window_size() - 1}));
        const auto sdpa_front = register_new_node<v1::Maximum>(
            register_new_node<v1::Subtract>(past_scalar, window_minus_one),
            zero_scalar);
        const auto resident_past = register_new_node<v1::Subtract>(past_scalar, sdpa_front);
        outputs.mask_past_seqlen = register_new_node<v0::Unsqueeze>(resident_past, zero_axis);
        outputs.bias_col_offset = register_new_node<v0::Unsqueeze>(sdpa_front, zero_axis);
    }

    return outputs;
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                      const ov::Output<ov::Node>& key,
                                                                      const ov::Output<ov::Node>& value,
                                                                      const ov::Output<ov::Node>& mask,
                                                                      const ov::Output<ov::Node>& scale,
                                                                      const ov::Output<ov::Node>& sink,
                                                                      bool is_causal,
                                                                      int64_t local_window_size) {
    const auto compressed = m_use_compressed_sdpa;
    ov::OutputVector inputs{query, compressed ? m_compressed_key : key, compressed ? m_compressed_value : value};
    if (mask.get_node()) {
        inputs.push_back(mask);
    }
    if (scale.get_node()) {
        inputs.push_back(scale);
    }
    if (sink.get_node()) {
        inputs.push_back(sink);
    }
    if (compressed) {
        inputs.push_back(m_key_scale);
        inputs.push_back(m_value_scale);
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    const auto value_order = m_transpose_v ? std::vector<int64_t>{0, 1, 3, 2} : order;
    const auto alignment = is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT
                                     : op::SDPA::CausalMaskAlignment::UPPER_LEFT;
    // GQA's -1 sentinel for "no window" maps to SDPA's 0 = disabled.
    const int64_t sdpa_window = (local_window_size >= 1) ? local_window_size : 0;
    std::shared_ptr<op::SDPA> sdpa;
    if (compressed) {
        sdpa = register_new_node<op::SDPA>(inputs,
                                           is_causal,
                                           order,
                                           order,
                                           value_order,
                                           order,
                                           m_quantization_attrs,
                                           ov::element::dynamic,
                                           alignment,
                                           sdpa_window);
    } else {
        sdpa = register_new_node<op::SDPA>(inputs,
                                           is_causal,
                                           order,
                                           order,
                                           value_order,
                                           order,
                                           ov::element::dynamic,
                                           alignment,
                                           sdpa_window);
    }
    return sdpa;
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_1d,
                                                                                const ov::Output<ov::Node>& past_seqlen,
                                                                                const ov::element::Type& compute_type,
                                                                                bool causal,
                                                                                int64_t local_window_size,
                                                                                const ov::Output<ov::Node>& external_bias,
                                                                                const ov::Output<ov::Node>& bias_col_offset,
                                                                                bool sliding_window_cache,
                                                                                float scale,
                                                                                bool has_sink) {
    // The kernel applies the window natively via is_causal + sliding_window_size (see make_sdpa),
    // so the explicit mask subgraph is unneeded whenever that path is reachable.
    if (causal && !external_bias.get_node() && scale == 0.0f && !has_sink &&
        (local_window_size == -1 || sliding_window_cache)) {
        return nullptr;
    }

    if (sliding_window_cache && !external_bias.get_node()) {
        const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
        const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        const auto one_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        const auto window_size = local_window_size - 1;
        const auto window =
            register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {window_size}));

        std::shared_ptr<ov::Node> key_positions =
            register_new_node<v4::Range>(zero_scalar, kv_len_scalar, one_scalar, ov::element::i64);
        key_positions = register_new_node<v0::Unsqueeze>(key_positions, zero);
        std::shared_ptr<ov::Node> query_positions =
            register_new_node<v4::Range>(zero_scalar, curr_seqlen_scalar, one_scalar, ov::element::i64);
        query_positions = register_new_node<v0::Unsqueeze>(query_positions, one);
        query_positions = register_new_node<v1::Add>(query_positions, past_seqlen);

        std::shared_ptr<ov::Node> masked = register_new_node<v1::Greater>(key_positions, query_positions);
        const auto distance = register_new_node<v1::Subtract>(query_positions, key_positions);
        const auto too_old = register_new_node<v1::Greater>(distance, window);
        masked = register_new_node<v1::LogicalOr>(masked, too_old);

        const auto typed_zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
        std::shared_ptr<ov::Node> minus_inf;
        if (compute_type == ov::element::f16) {
            minus_inf = register_new_node(
                v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
        } else if (compute_type == ov::element::bf16) {
            minus_inf = register_new_node(
                v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<ov::bfloat16>::lowest()}));
        } else {
            minus_inf = register_new_node(
                v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<float>::lowest()}));
        }
        auto mask = register_new_node<v1::Select>(masked, minus_inf, typed_zero);

        printf("Here statelessKV SWA mask(%d) on [%s]\n", local_window_size, curr_seqlen_scalar.get_node()->get_friendly_name().c_str());
        return mask;
    }

    return ov::pass::GroupQueryAttentionDecomposition::make_attention_mask(curr_seqlen_scalar,
                                                                           kv_len_scalar,
                                                                           kv_len_1d,
                                                                           past_seqlen,
                                                                           compute_type,
                                                                           causal,
                                                                           local_window_size,
                                                                           external_bias,
                                                                           bias_col_offset,
                                                                           sliding_window_cache,
                                                                           scale,
                                                                           has_sink);
}

}  // namespace ov::intel_gpu