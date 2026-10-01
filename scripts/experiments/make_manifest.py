"""Reproducibility manifest for the four experiments -> results/four_experiments_manifest.json."""

from __future__ import annotations

import json

from common import RESULTS, ROOT, environment, sha256_file, source_state, write_json  # noqa: E402

COMMANDS = [
    "export OMP_NUM_THREADS=1   # timing only; predictions are unaffected",
    "python scripts/experiments/exp1_2_multiseed_ablation.py --seeds 42 43 44 45 46 47 48 49 50 51",
    "python scripts/experiments/exp1_2_analysis.py",
    "python scripts/experiments/exp2b_template_disjoint.py --seeds 42 43 44 45 46 47 48 49 50 51",
    "python scripts/experiments/exp3_4_replay.py --seed 42 --n 60000 --db <scratch>/replay_seed42.db",
    "python scripts/experiments/exp3_e2e_analysis.py",
    "python scripts/experiments/exp4_replay_models.py --seed 42",
    "python scripts/experiments/exp4_rebuild_validation.py --seed 42 --sample 60 --final-db <scratch>/replay_seed42.db",
    "python scripts/experiments/leakage_audit.py",
    "python scripts/experiments/make_manifest.py",
]


def main() -> int:
    outputs = {}
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file() and "_sanity" not in p.parts and p.name != "four_experiments_manifest.json":
            outputs[str(p.relative_to(ROOT))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    scripts = {str(p.relative_to(ROOT)): sha256_file(p)
               for p in sorted((ROOT / "scripts" / "experiments").glob("*.py"))}
    scripts["backend/ml/replay_corpus.py"] = sha256_file(ROOT / "backend/ml/replay_corpus.py")
    models = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / "models").glob("*")) if p.is_file()}
    run = json.loads((RESULTS / "graph_replay/replay_run_seed42.json").read_text())
    write_json(RESULTS / "four_experiments_manifest.json", {
        "experiments": {
            "1_multiseed": {"seeds": list(range(42, 52)), "unit_of_replication": "seed",
                            "outputs": ["results/multiseed/multiseed_results.csv", "results/multiseed/multiseed_summary.json",
                                        "results/multiseed/seed_configurations.json"]},
            "2_ablation": {"configs": 11, "seeds": list(range(42, 52)),
                           "outputs": ["results/ablation/ablation_results.csv", "results/ablation/ablation_summary.json",
                                       "results/ablation/ablation_paired_deltas.csv",
                                       "results/ablation/template_disjoint_results.csv",
                                       "results/ablation/template_disjoint_summary.json"]},
            "3_end_to_end": {"assessment_function": "backend.app.services.payee_check.check_payee",
                             "outputs": ["results/end_to_end/e2e_results.csv", "results/end_to_end/e2e_summary.json",
                                         "results/end_to_end/evidence_family_metrics.csv",
                                         "results/end_to_end/e2e_audit_sample.jsonl",
                                         "results/end_to_end/e2e_audit_full_seed42.jsonl"]},
            "4_graph_replay": {"order": "assess at t from state < t, then record_payment",
                               "clock_injection": run.get("clock_injection"),
                               "outputs": ["results/graph_replay/replay_transactions_seed42.csv",
                                           "results/graph_replay/replay_evidence_seed42.csv",
                                           "results/graph_replay/replay_metrics_seed42.json",
                                           "results/graph_replay/replay_model_results_seed42.csv",
                                           "results/graph_replay/replay_validation.json",
                                           "results/graph_replay/replay_run_seed42.json"]},
        },
        "commands": COMMANDS,
        "runtime_substitutions": ["backend.app.core.clock.now_app -> replay time (replay only)",
                                  "re._MAXCACHE = 8192 (performance only)",
                                  "OMP_NUM_THREADS=1 (performance only)"],
        "replay_seconds": run.get("seconds"),
        "outputs": outputs, "experiment_scripts": scripts, "model_artefacts": models,
        "environment": environment(), "source": source_state(),
        "original_artefacts_untouched": ["models/", "evaluation/", "evaluation_v2/"],
    })
    print(len(outputs), "outputs hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
