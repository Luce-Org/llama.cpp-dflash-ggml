#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(condition) do {                                               \
    if (!(condition)) {                                                     \
        std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n",                 \
                     #condition, __FILE__, __LINE__);                       \
        std::abort();                                                       \
    }                                                                       \
} while (0)

static ggml_backend_meta_split_state split_state(
        const ggml_tensor * tensor,
        void * userdata) {
    const size_t n_devices = *static_cast<const size_t *>(userdata);
    ggml_backend_meta_split_state state{};
    if (std::strcmp(tensor->name, "repeated_axis2") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = 2;
        state.ne[1] = 2;
        state.nr[0] = 3;
        state.n_segments = 1;
        return state;
    }
    if (std::strcmp(tensor->name, "axis2") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = tensor->ne[2] / 2;
        state.ne[1] = tensor->ne[2] - state.ne[0];
        // Leave nr zero to cover callbacks compiled before repetition counts
        // were added to the split-state contract.
        state.n_segments = 1;
        return state;
    }
    if (std::strcmp(tensor->name, "row_weight") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.ne[0] = tensor->ne[0] / 2;
        state.ne[1] = tensor->ne[0] - state.ne[0];
        state.nr[0] = 1;
        state.n_segments = 1;
        return state;
    }
    if (std::strcmp(tensor->name, "column_weight") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_1;
        state.ne[0] = tensor->ne[1] / 2;
        state.ne[1] = tensor->ne[1] - state.ne[0];
        state.nr[0] = 1;
        state.n_segments = 1;
        return state;
    }
    if (std::strcmp(tensor->name, "repeated") != 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        state.nr[0] = 1;
        state.n_segments = 1;
        return state;
    }

    if (n_devices != 2) {
        std::fprintf(stderr, "test fixture requires exactly two devices\n");
        std::abort();
    }
    state.axis = GGML_BACKEND_SPLIT_AXIS_0;
    state.ne[0] = 2;
    state.ne[1] = 2;
    state.ne[2] = 4;
    state.ne[3] = 4;
    state.nr[0] = 2;
    state.nr[1] = 2;
    state.n_segments = 2;
    return state;
}

int main() {
    ggml_backend_load_all();

    const char * first_name = std::getenv("GGML_META_TEST_DEVICE_0");
    const char * second_name = std::getenv("GGML_META_TEST_DEVICE_1");
    first_name = first_name ? first_name : "CUDA0";
    second_name = second_name ? second_name : "CUDA1";

    ggml_backend_dev_t devices[] = {
        ggml_backend_dev_by_name(first_name),
        ggml_backend_dev_by_name(second_name),
    };
    if (!devices[0] || !devices[1]) {
        std::printf("SKIP: two requested devices are not available (%s,%s)\n",
                    first_name, second_name);
        return 0;
    }

    size_t n_devices = 2;
    ggml_backend_dev_t meta_device =
        ggml_backend_meta_device(devices, n_devices, split_state, &n_devices);
    CHECK(meta_device);
    ggml_backend_t backend = ggml_backend_dev_init(meta_device, nullptr);
    CHECK(backend);

    ggml_init_params params{};
    constexpr size_t graph_nodes = 64;
    params.mem_size = 8 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx);

    ggml_tensor * repeated = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 4);
    ggml_set_name(repeated, "repeated");
    ggml_tensor * mirrored = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 4);
    ggml_set_name(mirrored, "mirrored");
    ggml_tensor * repeated_axis2 =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 12, 2);
    ggml_set_name(repeated_axis2, "repeated_axis2");
    ggml_tensor * axis2 =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 12, 2);
    ggml_set_name(axis2, "axis2");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer);
    ggml_tensor * repeated_0 = ggml_backend_meta_simple_tensor(repeated, 0);
    ggml_tensor * repeated_1 = ggml_backend_meta_simple_tensor(repeated, 1);
    CHECK(repeated_0);
    CHECK(repeated_1);
    CHECK(repeated_0 != repeated_1);
    CHECK(ggml_backend_meta_simple_tensor(repeated, 2) == nullptr);
    CHECK(ggml_backend_meta_simple_tensor(nullptr, 0) == nullptr);
    CHECK(ggml_nelements(repeated_0) + ggml_nelements(repeated_1) ==
          ggml_nelements(repeated));

    std::vector<float> input((size_t) ggml_nelements(repeated));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (float) i + 0.25f;
    ggml_backend_tensor_set(repeated, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(mirrored, input.data(), 0, input.size() * sizeof(float));

    std::vector<float> output(input.size(), 0.0f);
    ggml_backend_tensor_get(repeated, output.data(), 0, output.size() * sizeof(float));
    CHECK(output == input);
    std::fill(output.begin(), output.end(), 0.0f);
    ggml_backend_tensor_get(mirrored, output.data(), 0, output.size() * sizeof(float));
    CHECK(output == input);

    std::vector<float> partial(48, 0.0f);
    ggml_backend_tensor_get(repeated, partial.data(), repeated->nb[1],
                            partial.size() * sizeof(float));
    for (size_t i = 0; i < partial.size(); ++i) {
        CHECK(partial[i] == input[i + 24]);
    }

    std::vector<float> axis2_input((size_t) ggml_nelements(repeated_axis2));
    for (size_t i = 0; i < axis2_input.size(); ++i) axis2_input[i] = (float) i + 0.5f;
    ggml_backend_tensor_set(repeated_axis2, axis2_input.data(), 0,
                            axis2_input.size() * sizeof(float));
    std::vector<float> axis2_output(axis2_input.size(), 0.0f);
    ggml_backend_tensor_get(repeated_axis2, axis2_output.data(), 0,
                            axis2_output.size() * sizeof(float));
    CHECK(axis2_output == axis2_input);

    std::vector<float> axis2_partial((size_t) repeated_axis2->ne[0] *
                                     repeated_axis2->ne[1] * repeated_axis2->ne[2]);
    ggml_backend_tensor_get(repeated_axis2, axis2_partial.data(),
                            repeated_axis2->nb[3], repeated_axis2->nb[3]);
    for (size_t i = 0; i < axis2_partial.size(); ++i) {
        CHECK(axis2_partial[i] == axis2_input[i + axis2_partial.size()]);
    }

    std::vector<float> simple_axis2_input((size_t) ggml_nelements(axis2));
    for (size_t i = 0; i < simple_axis2_input.size(); ++i) {
        simple_axis2_input[i] = (float) i + 0.75f;
    }
    ggml_backend_tensor_set(axis2, simple_axis2_input.data(), 0,
                            simple_axis2_input.size() * sizeof(float));
    std::vector<float> simple_axis2_output(simple_axis2_input.size(), 0.0f);
    ggml_backend_tensor_get(axis2, simple_axis2_output.data(), 0,
                            simple_axis2_output.size() * sizeof(float));
    CHECK(simple_axis2_output == simple_axis2_input);

    const int64_t test_head = 7;
    const size_t partial_bytes = 2 * axis2->nb[1];
    std::vector<float> simple_axis2_partial(
        partial_bytes / sizeof(float), 0.0f);
    ggml_backend_tensor_get(axis2, simple_axis2_partial.data(),
                            (size_t) test_head * axis2->nb[2],
                            partial_bytes);
    const size_t logical_start =
        (size_t) test_head * axis2->ne[0] * axis2->ne[1];
    for (size_t i = 0; i < simple_axis2_partial.size(); ++i) {
        CHECK(simple_axis2_partial[i] == simple_axis2_input[logical_start + i]);
        simple_axis2_partial[i] = -(float) i - 1.0f;
    }
    ggml_backend_tensor_set(axis2, simple_axis2_partial.data(),
                            (size_t) test_head * axis2->nb[2],
                            partial_bytes);
    ggml_backend_tensor_get(axis2, simple_axis2_output.data(), 0,
                            simple_axis2_output.size() * sizeof(float));
    for (size_t i = 0; i < simple_axis2_partial.size(); ++i) {
        CHECK(simple_axis2_output[logical_start + i] ==
               simple_axis2_partial[i]);
    }

    ggml_init_params weight_params{};
    weight_params.mem_size = 8 * ggml_tensor_overhead();
    weight_params.no_alloc = true;
    ggml_context * weight_ctx = ggml_init(weight_params);
    CHECK(weight_ctx);
    ggml_tensor * column_weight =
        ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_name(column_weight, "column_weight");
    ggml_tensor * row_weight =
        ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, 8, 4);
    ggml_set_name(row_weight, "row_weight");
    ggml_backend_buffer_t weight_buffer =
        ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
    CHECK(weight_buffer);
    ggml_backend_buffer_set_usage(weight_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_init_params graph_params{};
    graph_params.mem_size = 16 * ggml_tensor_overhead() +
                            ggml_graph_overhead_custom(graph_nodes, false);
    graph_params.no_alloc = true;
    ggml_context * graph_ctx = ggml_init(graph_params);
    CHECK(graph_ctx);
    ggml_tensor * mat_input =
        ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(mat_input, "mat_input");
    ggml_tensor * mat_hidden =
        ggml_mul_mat(graph_ctx, column_weight, mat_input);
    ggml_set_name(mat_hidden, "mat_hidden");
    ggml_tensor * mat_result = ggml_mul_mat(graph_ctx, row_weight, mat_hidden);
    ggml_set_name(mat_result, "mat_result");
    ggml_tensor * mat_output = ggml_scale(graph_ctx, mat_result, 1.0f);
    ggml_set_name(mat_output, "mat_output");
    ggml_set_output(mat_output);
    ggml_cgraph * graph =
        ggml_new_graph_custom(graph_ctx, graph_nodes, false);
    ggml_build_forward_expand(graph, mat_output);
    ggml_gallocr_t graph_alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(graph_alloc);
    CHECK(ggml_gallocr_alloc_graph(graph_alloc, graph));
    CHECK(ggml_backend_meta_simple_tensor(mat_output, 0));
    CHECK(ggml_backend_meta_simple_tensor(mat_output, 1));

    std::vector<float> column_data((size_t) ggml_nelements(column_weight));
    std::vector<float> row_data((size_t) ggml_nelements(row_weight));
    std::vector<float> mat_input_data((size_t) ggml_nelements(mat_input));
    for (size_t i = 0; i < column_data.size(); ++i) {
        column_data[i] = (float) ((int) (i % 11) - 5) / 8.0f;
    }
    for (size_t i = 0; i < row_data.size(); ++i) {
        row_data[i] = (float) ((int) (i % 13) - 6) / 9.0f;
    }
    for (size_t i = 0; i < mat_input_data.size(); ++i) {
        mat_input_data[i] = (float) ((int) (i % 7) - 3) / 4.0f;
    }
    ggml_backend_tensor_set(column_weight, column_data.data(), 0,
                            column_data.size() * sizeof(float));
    ggml_backend_tensor_set(row_weight, row_data.data(), 0,
                            row_data.size() * sizeof(float));
    ggml_backend_tensor_set(mat_input, mat_input_data.data(), 0,
                            mat_input_data.size() * sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result((size_t) ggml_nelements(mat_output));
    ggml_backend_tensor_get(mat_output, result.data(), 0,
                            result.size() * sizeof(float));
    std::vector<float> hidden(16, 0.0f);
    for (int column = 0; column < 2; ++column) {
        for (int row = 0; row < 8; ++row) {
            for (int k = 0; k < 4; ++k) {
                hidden[(size_t) column * 8 + row] +=
                    column_data[(size_t) row * 4 + k] *
                    mat_input_data[(size_t) column * 4 + k];
            }
        }
    }
    for (int column = 0; column < 2; ++column) {
        for (int row = 0; row < 4; ++row) {
            float expected = 0.0f;
            for (int k = 0; k < 8; ++k) {
                expected += row_data[(size_t) row * 8 + k] *
                            hidden[(size_t) column * 8 + k];
            }
            const float actual = result[(size_t) column * 4 + row];
            if (std::fabs(actual - expected) >= 1e-5f) {
                std::fprintf(stderr,
                    "matmul mismatch column=%d row=%d actual=%f expected=%f\n",
                    column, row, actual, expected);
            }
            CHECK(std::fabs(actual - expected) < 1e-5f);
        }
    }

    ggml_gallocr_free(graph_alloc);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("meta backend selected-device round trip passed (%s,%s)\n",
                first_name, second_name);
    return 0;
}
