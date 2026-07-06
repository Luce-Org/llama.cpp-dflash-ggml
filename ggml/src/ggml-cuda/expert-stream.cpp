#include "expert-stream.h"

#include <algorithm>
#include <unordered_map>

namespace ggml_moe_stream {

residency_planner::residency_planner(int n_experts, int budget, float decay, float hysteresis)
    : n_experts_(n_experts),
      budget_(std::min(budget, n_experts)),
      decay_(decay),
      hysteresis_(hysteresis),
      score_(n_experts, 0.0),
      resident_(n_experts, 0) {
    // Deterministic warm default: the lowest `budget` ids start resident.
    for (int e = 0; e < budget_; ++e) resident_[e] = 1;
}

void residency_planner::observe(const int32_t * expert_ids, int n) {
    if (decay_ != 1.0) {
        for (double & s : score_) s *= decay_;
    }
    for (int i = 0; i < n; ++i) {
        const int e = expert_ids[i];
        if (e >= 0 && e < n_experts_) score_[e] += 1.0;
    }
    update_residency();
}

// Swap the weakest resident for the strongest challenger while the challenger
// beats it by the hysteresis margin. Converges to (a sticky) top-`budget`.
void residency_planner::update_residency() {
    for (;;) {
        int    victim = -1;     double victim_score = 0;     // weakest resident
        int    chall  = -1;     double chall_score  = -1;    // strongest non-resident
        for (int e = 0; e < n_experts_; ++e) {
            if (resident_[e]) {
                if (victim == -1 || score_[e] < victim_score) { victim = e; victim_score = score_[e]; }
            } else {
                if (score_[e] > chall_score) { chall = e; chall_score = score_[e]; }
            }
        }
        if (victim == -1 || chall == -1) break;
        if (chall_score > victim_score * hysteresis_) {
            resident_[victim] = 0;
            resident_[chall]  = 1;
        } else {
            break;
        }
    }
}

std::vector<int> residency_planner::resident_set() const {
    std::vector<int> out;
    for (int e = 0; e < n_experts_; ++e) if (resident_[e]) out.push_back(e);
    return out; // already ascending
}

bool residency_planner::is_resident(int e) const {
    return e >= 0 && e < n_experts_ && resident_[e];
}

double residency_planner::usage(int e) const {
    return (e >= 0 && e < n_experts_) ? score_[e] : 0.0;
}

std::vector<int> plan_prefetch(const std::vector<int>  & predicted_experts,
                               const std::vector<uint8_t> & resident_mask,
                               int max_prefetch) {
    // demand-count cold (non-resident) experts; preserve first-seen for stability
    std::unordered_map<int, int> demand;
    std::vector<int>             order; // first-seen order of distinct cold ids
    for (int e : predicted_experts) {
        if (e < 0 || e >= (int) resident_mask.size() || resident_mask[e]) continue;
        if (demand.find(e) == demand.end()) order.push_back(e);
        demand[e]++;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (demand[a] != demand[b]) return demand[a] > demand[b]; // higher demand first
        return a < b;                                             // tie: lower id first
    });
    if (max_prefetch >= 0 && (int) order.size() > max_prefetch) {
        order.resize(max_prefetch);
    }
    return order;
}

} // namespace ggml_moe_stream
