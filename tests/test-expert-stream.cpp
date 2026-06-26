// Unit tests for the MoE expert-streaming policy layer (P1 prefetch, P3 residency).
// Pure CPU — no CUDA, no model. Compile: nvcc -std=c++17 test + expert-stream.cpp.

#include "expert-stream.h"

#include <cstdio>
#include <vector>
#include <algorithm>

using namespace ggml_moe_stream;

static int g_total = 0, g_fail = 0;
#define CHECK(cond) do { ++g_total; if (!(cond)) { ++g_fail; \
    std::printf("  FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static bool eq(const std::vector<int>& a, const std::vector<int>& b) { return a == b; }
static std::vector<int> sorted(std::vector<int> v) { std::sort(v.begin(), v.end()); return v; }

static void obs(residency_planner& p, std::initializer_list<int32_t> ids, int times = 1) {
    std::vector<int32_t> v(ids);
    for (int t = 0; t < times; ++t) p.observe(v.data(), (int) v.size());
}

// ----------------------------- P3: residency -------------------------------

static void test_p3_cold_start_defaults_to_low_ids() {
    // With no usage, resident set is deterministic: the lowest `budget` ids.
    residency_planner p(/*n*/8, /*budget*/3, /*decay*/1.0f, /*hyst*/1.10f);
    CHECK(eq(sorted(p.resident_set()), {0, 1, 2}));
    CHECK((int) p.resident_set().size() == 3);
}

static void test_p3_selects_hot_experts() {
    residency_planner p(8, 3, 1.0f, 1.10f);
    obs(p, {5, 7, 6}, 100);           // experts 5,6,7 hammered
    CHECK(eq(sorted(p.resident_set()), {5, 6, 7}));
}

static void test_p3_budget_never_exceeded() {
    residency_planner p(16, 4, 1.0f, 1.10f);
    for (int e = 0; e < 16; ++e) obs(p, {e}, e + 1); // varied frequencies
    CHECK((int) p.resident_set().size() == 4);
    int cnt = 0; for (auto m : p.resident_mask()) cnt += m;
    CHECK(cnt == 4);
}

static void test_p3_recency_shifts_resident_set() {
    // Hot early then cold; decay should let a later-hot group take over.
    residency_planner p(32, 3, /*decay*/0.99f, /*hyst*/1.10f);
    obs(p, {1, 2, 3}, 1000);
    CHECK(eq(sorted(p.resident_set()), {1, 2, 3}));
    obs(p, {20, 21, 22}, 1000);
    CHECK(eq(sorted(p.resident_set()), {20, 21, 22}));
}

static void test_p3_hysteresis_prevents_thrash() {
    // budget 1, two experts. 0 is hot (score 100). 1 must beat 100*1.10 = 110.
    residency_planner p(2, 1, /*decay*/1.0f, /*hyst*/1.10f);
    obs(p, {0}, 100);
    CHECK(eq(p.resident_set(), {0}));
    obs(p, {1}, 105);               // score 105 < 110: no displacement
    CHECK(eq(p.resident_set(), {0}));
    obs(p, {1}, 6);                 // score 111 > 110: now displaces
    CHECK(eq(p.resident_set(), {1}));
}

// ----------------------------- P1: prefetch --------------------------------

static void test_p1_only_cold_experts() {
    std::vector<uint8_t> resident(8, 0); resident[2] = 1; // expert 2 resident
    auto plan = plan_prefetch({1, 2, 3}, resident, /*max*/-1);
    CHECK(eq(sorted(plan), {1, 3}));        // 2 skipped (already resident)
}

static void test_p1_dedup() {
    std::vector<uint8_t> resident(8, 0);
    auto plan = plan_prefetch({1, 1, 1, 3, 3}, resident, -1);
    CHECK(eq(sorted(plan), {1, 3}));        // each prefetched once
}

static void test_p1_priority_by_demand() {
    std::vector<uint8_t> resident(8, 0);
    // 5 needed 3x, 2 once, 7 once -> 5 first; ties (2,7) by id asc.
    auto plan = plan_prefetch({5, 5, 5, 2, 7}, resident, -1);
    CHECK(eq(plan, {5, 2, 7}));
}

static void test_p1_bandwidth_cap() {
    std::vector<uint8_t> resident(16, 0);
    // demand: 5(x3), 9(x2), 2(x1), 7(x1). cap 2 -> top two by demand: 5, 9.
    auto plan = plan_prefetch({5, 5, 5, 9, 9, 2, 7}, resident, /*max*/2);
    CHECK(eq(plan, {5, 9}));
}

static void test_p1_all_resident_empty_plan() {
    std::vector<uint8_t> resident(8, 0); resident[1] = 1; resident[2] = 1;
    auto plan = plan_prefetch({1, 2}, resident, -1);
    CHECK(plan.empty());                    // the win: nothing to stream
}

int main() {
    test_p3_cold_start_defaults_to_low_ids();
    test_p3_selects_hot_experts();
    test_p3_budget_never_exceeded();
    test_p3_recency_shifts_resident_set();
    test_p3_hysteresis_prevents_thrash();

    test_p1_only_cold_experts();
    test_p1_dedup();
    test_p1_priority_by_demand();
    test_p1_bandwidth_cap();
    test_p1_all_resident_empty_plan();

    std::printf("\n%d/%d checks passed (%d failed)\n", g_total - g_fail, g_total, g_fail);
    return g_fail ? 1 : 0;
}
