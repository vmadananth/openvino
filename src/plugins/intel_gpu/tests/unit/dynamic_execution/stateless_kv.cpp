// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/stateless_kv.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "stateless_kv_inst.h"

using namespace cldnn;
using namespace ::tests;

struct stateless_kv_runtime_params {
    int64_t new_token_len;
    int64_t seq_len;
    bool is_seq_len_present_len;
    bool has_pos_idx;
};

class stateless_kv_runtime : public testing::TestWithParam<stateless_kv_runtime_params> {};

TEST_P(stateless_kv_runtime, output_memory_reuse) {
    const auto& params = GetParam();
    auto& engine = get_test_engine();

    constexpr int64_t past_capacity = 16;
    constexpr int64_t batch = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t head_size = 4;
    const auto logical_past_len = params.is_seq_len_present_len ? params.seq_len - params.new_token_len : params.seq_len;
    const auto logical_present_len = params.is_seq_len_present_len ? params.seq_len : params.seq_len + params.new_token_len;
    const auto output_capacity = std::max(past_capacity, logical_present_len);
    const bool reuse_past_buffer = logical_present_len <= past_capacity;
    const auto past_layout = layout{ov::PartialShape{batch, heads, past_capacity, head_size}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{batch, heads, params.new_token_len, head_size}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};
    const auto pos_idx_layout = layout{ov::PartialShape{params.new_token_len}, data_types::i32, format::bfyx};

    std::vector<input_info> stateless_kv_inputs = {
        input_info("past"),
        input_info("new_token"),
        input_info("present_len"),
    };
    if (params.has_pos_idx) {
        stateless_kv_inputs.emplace_back("pos_idx");
    }
    auto stateless_kv_prim = stateless_kv("stateless_kv", stateless_kv_inputs, 2, params.is_seq_len_present_len);
    stateless_kv_prim.num_outputs = 2;
    stateless_kv_prim.output_data_types = {data_types::f32, data_types::f32};

    topology topology;
    topology.add(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}));
    topology.add(input_layout("new_token", new_token_layout));
    topology.add(input_layout("present_len", seq_len_layout));
    if (params.has_pos_idx) {
        topology.add(input_layout("pos_idx", pos_idx_layout));
    }
    topology.add(stateless_kv_prim);
    topology.add(reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto past = engine.allocate_memory(past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    auto pos_idx = engine.allocate_memory(pos_idx_layout);
    std::vector<float> past_values(past_layout.count());
    std::vector<float> new_token_values(new_token_layout.count());
    for (size_t i = 0; i < past_values.size(); ++i) {
        past_values[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < new_token_values.size(); ++i) {
        new_token_values[i] = static_cast<float>(1000 + i);
    }
    set_values(past, past_values);
    set_values(new_token, new_token_values);
    set_values<int64_t>(seq_len, {params.seq_len});
    std::vector<int32_t> pos_idx_values(static_cast<size_t>(params.new_token_len));
    std::iota(pos_idx_values.begin(), pos_idx_values.end(), static_cast<int32_t>(logical_past_len));
    set_values(pos_idx, pos_idx_values);

    std::vector<float> expected(static_cast<size_t>(batch * heads * output_capacity * head_size));
    for (int64_t head = 0; head < heads; ++head) {
        const auto past_head_offset = head * past_capacity * head_size;
        const auto output_head_offset = head * output_capacity * head_size;
        const auto token_head_offset = head * params.new_token_len * head_size;
        std::copy_n(past_values.begin() + past_head_offset, past_capacity * head_size, expected.begin() + output_head_offset);
        std::copy_n(new_token_values.begin() + token_head_offset,
                    params.new_token_len * head_size,
                    expected.begin() + output_head_offset + logical_past_len * head_size);
    }

    network.set_input_data("past", past);
    network.set_input_data("new_token", new_token);
    network.set_input_data("present_len", seq_len);
    if (params.has_pos_idx) {
        network.set_input_data("pos_idx", pos_idx);
    }
    network.set_output_memory("result", past);

    const auto outputs = network.execute();
    const auto stateless_kv_inst = network.get_primitive("stateless_kv");
    const auto output0 = stateless_kv_inst->output_memory_ptr(0);
    const auto output1 = stateless_kv_inst->output_memory_ptr(1);
    const auto result = outputs.at("result").get_memory();

    ASSERT_NE(output0, nullptr);
    ASSERT_NE(output1, nullptr);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*output0, *result));
    EXPECT_TRUE(engine.is_the_same_buffer(*output1, *output0));
    EXPECT_EQ(engine.is_the_same_buffer(*output0, *past), reuse_past_buffer);
    EXPECT_EQ(output0->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, output_capacity, head_size}));
    EXPECT_EQ(output1->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, logical_present_len, head_size}));
    ASSERT_NE(output0->get_mem_tracker(), nullptr);
    ASSERT_GE(output0->get_mem_tracker()->size(), output0->get_layout().bytes_count());

    mem_lock<float, mem_lock_type::read> output_lock(output0, get_test_stream());
    const std::vector<float> actual(output_lock.begin(), output_lock.begin() + expected.size());
    EXPECT_THAT(actual, testing::ElementsAreArray(expected));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         stateless_kv_runtime,
                         testing::Values(stateless_kv_runtime_params{2, 15, true, false},
                                         stateless_kv_runtime_params{1, 16, true, false},
                                         stateless_kv_runtime_params{2, 18, true, false},
                                         stateless_kv_runtime_params{2, 13, false, false},
                                         stateless_kv_runtime_params{1, 15, false, false},
                                         stateless_kv_runtime_params{2, 15, false, false},
                                         stateless_kv_runtime_params{2, 16, false, false},
                                         stateless_kv_runtime_params{2, 15, true, true},
                                         stateless_kv_runtime_params{1, 16, true, true},
                                         stateless_kv_runtime_params{2, 18, true, true},
                                         stateless_kv_runtime_params{2, 13, false, true},
                                         stateless_kv_runtime_params{1, 15, false, true},
                                         stateless_kv_runtime_params{2, 16, false, true}),
                         [](const testing::TestParamInfo<stateless_kv_runtime_params>& info) {
                             const auto& params = info.param;
                             return std::string{params.is_seq_len_present_len ? "PresentSeq" : "PastSeq"} + std::to_string(params.seq_len) + "_NewToken" +
                                    std::to_string(params.new_token_len) + (params.has_pos_idx ? "_Scatter" : "_Concat");
                         });

struct stateless_kv_swa_runtime_params : stateless_kv_runtime_params {
    stateless_kv_swa_runtime_params(int64_t new_token_len, int64_t past_seq_len, bool reuse_present_buffer)
        : stateless_kv_runtime_params{new_token_len, past_seq_len, false, false},
          reuse_present_buffer(reuse_present_buffer) {}

    bool reuse_present_buffer;
};

class stateless_kv_swa_runtime : public testing::TestWithParam<stateless_kv_swa_runtime_params> {};

TEST_P(stateless_kv_swa_runtime, swa_memory) {
    const auto& params = GetParam();
    auto& engine = get_test_engine();

    constexpr int64_t past_capacity = 16;
    constexpr int64_t batch = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t head_size = 4;
    constexpr int64_t window_size = 4;
    ASSERT_FALSE(params.has_pos_idx);
    ASSERT_FALSE(params.is_seq_len_present_len);

    const auto past_seq_len = params.seq_len;
    const auto present_seq_len = past_seq_len + params.new_token_len;
    const auto get_front_idx = [&](int64_t seq_len) {
        if (seq_len <= past_capacity)
            return int64_t{0};
        const auto rolling_gap = past_capacity - window_size + 1;
        return rolling_gap * (1 + (seq_len - past_capacity - 1) / rolling_gap);
    };
    const auto past_front_idx = get_front_idx(past_seq_len);
    const auto present_front_idx = get_front_idx(present_seq_len);
    const auto is_rolling = present_front_idx > past_front_idx;
    const auto past_resident_len = past_seq_len - past_front_idx;
    const auto present_resident_len = present_seq_len - present_front_idx;
    const auto sdpa_front_idx = std::max<int64_t>(0, past_seq_len - window_size + 1);
    const auto sdpa_seq_len = present_seq_len - sdpa_front_idx;

    const auto make_token_values = [&](int64_t first_token_id, int64_t token_count, int64_t capacity) {
        std::vector<float> values(static_cast<size_t>(batch * heads * capacity * head_size), -1.0f);
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t token = 0; token < token_count; ++token) {
                const auto offset = (head * capacity + token) * head_size;
                std::fill_n(values.begin() + offset, head_size, static_cast<float>(first_token_id + token));
            }
        }
        return values;
    };
    const auto read_token_values = [&](const memory::ptr& output, int64_t token_count) {
        const auto& output_layout = output->get_layout();
        const auto pitches = output_layout.get_pitches();
        const auto base_offset = output_layout.get_linear_offset();
        mem_lock<float, mem_lock_type::read> output_lock(output, get_test_stream());
        std::vector<float> values;
        values.reserve(static_cast<size_t>(batch * heads * token_count * head_size));
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t token = 0; token < token_count; ++token) {
                for (int64_t element = 0; element < head_size; ++element) {
                    values.push_back(output_lock[base_offset + head * pitches[1] + token * pitches[2] + element * pitches[3]]);
                }
            }
        }
        return values;
    };

    const auto past_layout = layout{ov::PartialShape{batch, heads, past_capacity, head_size}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{batch, heads, params.new_token_len, head_size}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto stateless_kv_prim =
        stateless_kv("stateless_kv", {input_info("past"), input_info("new_token"), input_info("seq_len")}, 2, params.is_seq_len_present_len, window_size);
    stateless_kv_prim.num_outputs = 2;
    stateless_kv_prim.output_data_types = {data_types::f32, data_types::f32};

    topology topology(input_layout("past", layout{ov::PartialShape{batch, heads, -1, head_size}, data_types::f32, format::bfyx}),
                      input_layout("new_token", new_token_layout),
                      input_layout("seq_len", seq_len_layout),
                      stateless_kv_prim,
                      reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto past = engine.allocate_memory(past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    set_values(past, make_token_values(past_front_idx, past_resident_len, past_capacity));
    set_values(new_token, make_token_values(past_seq_len, params.new_token_len, params.new_token_len));
    set_values<int64_t>(seq_len, {params.seq_len});
    network.set_input_data("past", past);
    network.set_input_data("new_token", new_token);
    network.set_input_data("seq_len", seq_len);
    network.set_output_memory("result", past);

    const auto outputs = network.execute();
    const auto stateless_kv_inst = network.get_primitive("stateless_kv");
    const auto present = stateless_kv_inst->output_memory_ptr(0);
    const auto sdpa = stateless_kv_inst->output_memory_ptr(1);
    const auto result = outputs.at("result").get_memory();

    ASSERT_NE(present, nullptr);
    ASSERT_NE(sdpa, nullptr);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*present, *result));
    EXPECT_TRUE(engine.is_the_same_buffer(*present, *past));
    ASSERT_EQ(params.reuse_present_buffer, !is_rolling);
    EXPECT_EQ(engine.is_the_same_buffer(*sdpa, *present), params.reuse_present_buffer);
    ASSERT_EQ(present->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, past_capacity, head_size}));
    ASSERT_EQ(sdpa->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, sdpa_seq_len, head_size}));

    const auto present_ref = make_token_values(present_front_idx, present_resident_len, present_resident_len);
    const auto sdpa_ref = make_token_values(sdpa_front_idx, sdpa_seq_len, sdpa_seq_len);
    EXPECT_THAT(read_token_values(present, present_resident_len), testing::ElementsAreArray(present_ref));
    EXPECT_THAT(read_token_values(sdpa, sdpa_seq_len), testing::ElementsAreArray(sdpa_ref));
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    stateless_kv_swa_runtime,
    testing::Values(
        stateless_kv_swa_runtime_params{1, 14, true},   // no previous overflow, no rolling
        stateless_kv_swa_runtime_params{1, 16, false},  // no previous overflow, token 0 rolls
        stateless_kv_swa_runtime_params{3, 16, false},  // no previous overflow, token 0 rolls
        stateless_kv_swa_runtime_params{3, 15, false},  // no previous overflow, token 1 rolls
        stateless_kv_swa_runtime_params{1, 17, true},   // previous overflow, no rolling
        stateless_kv_swa_runtime_params{1, 29, false},  // previous overflow, token 0 rolls
        stateless_kv_swa_runtime_params{3, 29, false},  // previous overflow, token 0 rolls
        stateless_kv_swa_runtime_params{3, 28, false},  // previous overflow, token 1 rolls
        stateless_kv_swa_runtime_params{16, 29, false}  // block crosses multiple eviction boundaries
        ),
    [](const testing::TestParamInfo<stateless_kv_swa_runtime_params>& info) {
        const auto& params = info.param;
        return std::string{"Past"} + std::to_string(params.seq_len) + "_NewToken" + std::to_string(params.new_token_len) +
             (params.reuse_present_buffer ? "_ReusePresent" : "_SeparateSDPA");
    });

TEST(stateless_kv_runtime, swa_runtime_state_refresh) {
    auto& engine = get_test_engine();

    constexpr int64_t past_capacity = 16;
    constexpr int64_t window_size = 4;
    const auto past_layout = layout{ov::PartialShape{1, 2, past_capacity, 4}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{1, 2, 1, 4}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto primitive = stateless_kv("stateless_kv",
                                  {input_info("past"), input_info("new_token"), input_info("past_len")},
                                  2,
                                  false,
                                  window_size);
    primitive.num_outputs = 2;
    primitive.output_data_types = {data_types::f32, data_types::f32};

    topology topology(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}),
                      input_layout("new_token", new_token_layout),
                      input_layout("past_len", seq_len_layout),
                      primitive,
                      reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto past = engine.allocate_memory(past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto past_len = engine.allocate_memory(seq_len_layout);
    set_values(past, std::vector<float>(past_layout.count(), 0.0f));
    set_values(new_token, std::vector<float>(new_token_layout.count(), 1.0f));
    network.set_input_data("past", past);
    network.set_input_data("new_token", new_token);
    network.set_input_data("past_len", past_len);
    network.set_output_memory("result", past);

    const auto run_and_check = [&](int64_t absolute_past_len, bool expected_rolling) {
        set_values<int64_t>(past_len, {absolute_past_len});
        network.execute();

        const auto instance = std::dynamic_pointer_cast<cldnn::stateless_kv_inst>(network.get_primitive("stateless_kv"));
        ASSERT_NE(instance, nullptr);
        ASSERT_TRUE(instance->get_runtime_state().has_value());
        const auto& state = *instance->get_runtime_state();
        EXPECT_EQ(state.past_seq_len, absolute_past_len);
        EXPECT_EQ(state.present_seq_len, absolute_past_len + 1);
        EXPECT_EQ(state.new_token_len, 1);
        EXPECT_EQ(state.is_rolling(), expected_rolling);
        EXPECT_EQ(engine.is_the_same_buffer(*instance->output_memory_ptr(0), *instance->output_memory_ptr(1)),
                  !expected_rolling);
    };

    run_and_check(14, false);
    run_and_check(16, true);
}

TEST(stateless_kv_runtime, reallocation_across_executions) {
    auto& engine = get_test_engine();

    constexpr int64_t batch = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t initial_capacity = 16;
    constexpr int64_t new_token_len = 2;
    constexpr int64_t head_size = 4;
    const auto initial_past_layout = layout{ov::PartialShape{batch, heads, initial_capacity, head_size}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{batch, heads, new_token_len, head_size}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto stateless_kv_prim = stateless_kv("stateless_kv", {input_info("past"), input_info("new_token"), input_info("present_len")}, 2, true);
    stateless_kv_prim.num_outputs = 2;
    stateless_kv_prim.output_data_types = {data_types::f32, data_types::f32};

    topology topology(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}),
                      input_layout("new_token", new_token_layout),
                      input_layout("present_len", seq_len_layout),
                      stateless_kv_prim,
                      reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto current_past = engine.allocate_memory(initial_past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    std::vector<float> current_values(initial_past_layout.count());
    for (size_t i = 0; i < current_values.size(); ++i) {
        current_values[i] = static_cast<float>(i);
    }
    set_values(current_past, current_values);
    network.set_output_memory("result", current_past);

    int64_t current_capacity = initial_capacity;
    auto run_and_verify = [&](int64_t present_len, float token_base, bool expect_reuse) {
        const auto logical_past_len = present_len - new_token_len;
        const auto output_capacity = std::max(current_capacity, present_len);
        std::vector<float> new_token_values(new_token_layout.count());
        for (size_t i = 0; i < new_token_values.size(); ++i) {
            new_token_values[i] = token_base + static_cast<float>(i);
        }
        set_values(new_token, new_token_values);
        set_values<int64_t>(seq_len, {present_len});

        std::vector<float> expected(static_cast<size_t>(batch * heads * output_capacity * head_size));
        for (int64_t head = 0; head < heads; ++head) {
            const auto past_head_offset = head * current_capacity * head_size;
            const auto output_head_offset = head * output_capacity * head_size;
            const auto token_head_offset = head * new_token_len * head_size;
            std::copy_n(current_values.begin() + past_head_offset, current_capacity * head_size, expected.begin() + output_head_offset);
            std::copy_n(new_token_values.begin() + token_head_offset,
                        new_token_len * head_size,
                        expected.begin() + output_head_offset + logical_past_len * head_size);
        }

        network.set_input_data("past", current_past);
        network.set_input_data("new_token", new_token);
        network.set_input_data("present_len", seq_len);

        const auto outputs = network.execute();
        const auto stateless_kv_inst = network.get_primitive("stateless_kv");
        const auto output0 = stateless_kv_inst->output_memory_ptr(0);
        const auto output1 = stateless_kv_inst->output_memory_ptr(1);
        const auto result = outputs.at("result").get_memory();

        ASSERT_NE(output0, nullptr);
        ASSERT_NE(output1, nullptr);
        ASSERT_NE(result, nullptr);
        EXPECT_TRUE(engine.is_the_same_buffer(*output0, *result));
        EXPECT_TRUE(engine.is_the_same_buffer(*output1, *output0));
        EXPECT_EQ(engine.is_the_same_buffer(*output0, *current_past), expect_reuse);
        EXPECT_EQ(output0->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, output_capacity, head_size}));
        EXPECT_EQ(output1->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, present_len, head_size}));
        ASSERT_NE(output0->get_mem_tracker(), nullptr);
        ASSERT_GE(output0->get_mem_tracker()->size(), output0->get_layout().bytes_count());

        mem_lock<float, mem_lock_type::read> output_lock(output0, get_test_stream());
        const std::vector<float> actual(output_lock.begin(), output_lock.begin() + expected.size());
        EXPECT_THAT(actual, testing::ElementsAreArray(expected));

        current_past = output0;
        current_values = std::move(expected);
        current_capacity = output_capacity;
    };

    run_and_verify(15, 1000.0f, true);
    run_and_verify(18, 2000.0f, false);
    run_and_verify(17, 3000.0f, true);
    run_and_verify(20, 4000.0f, false);
}