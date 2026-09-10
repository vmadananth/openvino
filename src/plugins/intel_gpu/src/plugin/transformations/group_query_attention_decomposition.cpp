// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "group_query_attention_decomposition.hpp"

#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::intel_gpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;

GroupQueryAttentionDecomposition::KVCacheOutputs GroupQueryAttentionDecomposition::construct_kvcache(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
    const ov::Output<ov::Node>& past_key,
    const ov::Output<ov::Node>& past_value,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& seqlens_1d,
    const ov::Output<ov::Node>& past_seqlen,
    const ov::Output<ov::Node>& /*current_seqlen_scalar*/) {
    const auto window_size = node->get_sliding_window_cache() ? node->get_local_window_size() : 0;
    const auto key_cache = register_new_node<op::StatelessKV>(past_key, key, seqlens_1d, 2, true, window_size);
    const auto value_cache = register_new_node<op::StatelessKV>(past_value, value, seqlens_1d, 2, true, window_size);
    printf("Here statelessKV[%s][%s](%d) on [%s]\n",
           past_key.get_node()->get_friendly_name().c_str(),
           past_value.get_node()->get_friendly_name().c_str(),
           window_size,
           seqlens_1d.get_node()->get_friendly_name().c_str());

    KVCacheOutputs outputs;
    outputs.present_key = key_cache->output(0);
    outputs.present_value = value_cache->output(0);
    outputs.sdpa_key = key_cache->output(1);
    outputs.sdpa_value = value_cache->output(1);
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
                                                                      bool is_causal) {
    ov::OutputVector inputs{query, key, value};
    if (mask.get_node()) {
        inputs.push_back(mask);
    }
    if (scale.get_node()) {
        inputs.push_back(scale);
    }
    if (sink.get_node()) {
        inputs.push_back(sink);
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    auto sdpa = register_new_node<op::SDPA>(inputs,
                                       is_causal,
                                       order,
                                       order,
                                       order,
                                       order,
                                       ov::element::dynamic,
                                       is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT : op::SDPA::CausalMaskAlignment::UPPER_LEFT);
    if (m_local_window_size >= 1) {
        sdpa->set_sliding_window_size(m_local_window_size);
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
    m_local_window_size = local_window_size;

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

    if (causal && local_window_size == -1 && !sliding_window_cache && !external_bias.get_node() && scale == 0.0f && !has_sink) {
        printf("Here causal skip mask on [%s]\n", curr_seqlen_scalar.get_node()->get_friendly_name().c_str());
        return nullptr;
    }

    printf("Here original mask %c(%d) on [%s]\n",
           sliding_window_cache ? 'Y' : 'N',
           local_window_size,
           curr_seqlen_scalar.get_node()->get_friendly_name().c_str());
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