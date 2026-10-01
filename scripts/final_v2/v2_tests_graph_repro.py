"""Table 10 (software tests), Table 13 (graph statistics) and REPRODUCIBILITY_MANIFEST.md.

The test counts come from the authoritative run on the project machine
(python -m pytest tests/<module>.py -q --tb=no, one module at a time;
results/final_v2/software_tests/pytest_summary_by_file.csv). Categories are
the primary subject of each module; keyword counts are grep counts over the
test sources and show where a concern is exercised across modules.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "final_v2"

CATEGORY = {
    "Graph behaviour": ["test_graph_cache", "test_graph_semantics", "test_fraud_graph", "test_graph_detector"],
    "Risk aggregation, corroboration and verdicts": ["test_evidence_model", "test_fact_corroboration", "test_payee_check",
                                                     "test_score_granularity", "test_verdict_vocabulary", "test_risk_rules",
                                                     "test_open_findings"],
    "Missing / unavailable evidence": ["test_evidence_availability", "test_entity_status"],
    "Model, thresholding and evaluation pipeline": ["test_ml_family", "test_model_pipeline", "test_model_portability",
                                                    "test_evaluation_pipeline", "test_evaluation_v2"],
    "Feature construction (V2 dataset and serving)": ["test_features_v2", "test_dataset_v2", "test_serving_velocity",
                                                      "test_amount_context", "test_upi_amount_limits"],
    "Chronological ordering and time handling": ["test_cross_source_and_timezone", "test_statement_time", "test_history_rebuild",
                                                 "test_payee_history"],
    "JSON serialisation": ["test_predict_serialization"],
    "API behaviour and security boundary": ["test_api_contract", "test_api_boundary_security", "test_chat"],
    "Contextual families (message, link, address/QR)": ["test_coercion", "test_scam_link", "test_upi_identity"],
    "Statement ingestion and parsing": ["test_bank_statement_providers", "test_statement_ingestion", "test_statement_parser",
                                        "test_statement_pipeline", "test_parser_regressions", "test_real_bank_statements"],
    "Regression and miscellaneous edge cases": ["test_audit_regressions", "test_device_risk", "test_geo_risk"],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md(df, path, title, note=""):
    path.write_text(f"**{title}**\n\n" + df.to_markdown(index=False) + (f"\n\n{note}\n" if note else "\n"), encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> int:
    t = pd.read_csv(R / "software_tests/pytest_summary_by_file.csv")
    mapped = {m for ms in CATEGORY.values() for m in ms}
    assert mapped == set(t.file), (mapped ^ set(t.file))
    rows = []
    for cat, mods in CATEGORY.items():
        g = t[t.file.isin(mods)]
        rows.append({"Category (module subject)": cat, "Modules": len(g), "Tests": int(g.passed.sum() + g.failed.sum() + g.skipped.sum()),
                     "Passed": int(g.passed.sum()), "Failed": int(g.failed.sum()), "Skipped": int(g.skipped.sum()),
                     "Time (s)": round(float(g.seconds.sum()), 1), "Modules list": ", ".join(mods)})
    rows.append({"Category (module subject)": "Total", "Modules": len(t), "Tests": int(t.passed.sum() + t.failed.sum() + t.skipped.sum()),
                 "Passed": int(t.passed.sum()), "Failed": int(t.failed.sum()), "Skipped": int(t.skipped.sum()),
                 "Time (s)": round(float(t.seconds.sum()), 1), "Modules list": ""})
    t10 = pd.DataFrame(rows)
    src = {p.name: p.read_text() for p in (ROOT / "tests").glob("test_*.py")}
    kw = []
    for label, pat in (("thresholds", r"threshold"), ("missing / unavailable evidence", r"missing|unavailable"),
                       ("graph / components", r"graph|component"), ("verdicts", r"verdict"), ("corroboration", r"corroborat"),
                       ("strict time boundary (before=)", r"before="), ("JSON", r"\bjson\b"), ("serialisation", r"serial")):
        files = [n for n, s in src.items() if re.search(pat, s, re.I)]
        kw.append({"Concern": label, "Modules mentioning it": len(files),
                   "Mentions": sum(len(re.findall(pat, s, re.I)) for s in src.values())})
    md(t10, R / "tables/table10_software_tests.md", "Table 10 — Software test validation (pytest, project machine)",
       "Run on the project machine (Python 3.10.12, Linux 6.8 x86_64) with `python -m pytest tests/<module>.py -q --tb=no`, one module at a time, "
       "with a throwaway JWT_SECRET; conftest.py isolates every run in a temporary database. Six pytest warnings, no errors. "
       "Time = sum of pytest-reported module times. These are implementation-correctness results, NOT fraud-detection accuracy.")
    pd.DataFrame(kw).to_csv(R / "software_tests/test_keyword_coverage.csv", index=False)
    (R / "software_tests/software_tests_summary.json").write_text(json.dumps({
        "modules": int(len(t)), "tests": int(t.passed.sum()), "passed": int(t.passed.sum()), "failed": int(t.failed.sum()),
        "skipped": int(t.skipped.sum()), "warnings": 6, "pytest_time_s": round(float(t.seconds.sum()), 1),
        "collected_total": 820, "environment": "project machine, Python 3.10.12, pytest 9.1.1, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3",
        "keyword_coverage": kw}, indent=2))

    # Table 13 graph statistics
    gs = json.loads((R / "graph/graph_statistics.json").read_text())
    rows = []
    for s, g in gs.items():
        a = g["assessment_time"]
        rows.append({"Stream": s, "Payments observed": g["payments_observed"], "Payer nodes": g["payer_nodes"], "Payee nodes": g["payee_nodes"],
                     "Payer–payee edges": g["edges_payer_payee_pairs"], "Repeat edges (≥2 payments)": g["repeat_relationships_edges_with_2plus_payments"],
                     "Single-payment edges": g["single_payment_relationships"],
                     "Payee in-degree mean / median / max": f"{g['payee_in_degree_distinct_payers']['mean']:.2f} / {g['payee_in_degree_distinct_payers']['median']:.0f} / {g['payee_in_degree_distinct_payers']['max']}",
                     "Payer out-degree mean / median / max": f"{g['payer_out_degree_distinct_payees']['mean']:.2f} / {g['payer_out_degree_distinct_payees']['median']:.0f} / {g['payer_out_degree_distinct_payees']['max']}",
                     "Connected components": g["components_in_store_union_find"], "Largest component (nodes)": g["largest_component_store"],
                     "Components = independent rebuild": g["components_agree"], "Aggregates = edges": g["cache_payer_count_matches_edges"],
                     "Payees ≥10 payers": g["payees_with_10plus_payers"],
                     "Graph evidence available at assessment (%)": round(100 * a["graph_evidence_available_rate"], 2),
                     "Positional finding fired (fraud / legit)": f"{a['positional_finding_fired']['fraud']} / {a['positional_finding_fired']['legitimate']}",
                     "Payments with shape findings": a["rows_with_graph_shape_findings"]})
    t13 = pd.DataFrame(rows)
    md(t13, R / "tables/table13_graph_statistics.md", "Table 13 — Replayed payer–payee graph (end of each 150-day stream) and graph evidence at assessment time",
       "Explicit incremental graph maintained by the running store (payee_payers edges, payee_graph_cache aggregates, graph_components union–find). "
       "No graph neural network and no embeddings exist in the implementation. Component counts are cross-checked by an independent union–find "
       "built from the transaction file.")
    rel = []
    for s, g in gs.items():
        for lab, d in g["assessment_time"]["relationship_state_by_label"].items():
            rel.append({"Stream": s, "Label": "fraud" if lab == "1" else "legitimate", **d})
    md(pd.DataFrame(rel).fillna(0), R / "tables/table13b_relationship_state.md", "Table 13b — Relationship state at assessment time (payee_status)")

    # reproducibility manifest
    import numpy, pandas, sklearn, scipy, joblib, matplotlib
    try:
        import shap; shv = shap.__version__
    except Exception:
        shv = "not installed"
    frozen = json.loads((ROOT / "models/metrics_v2.json").read_text())
    arts = {p: sha(ROOT / p) for p in ["models/risk_model_v2.pkl", "models/metrics_v2.json", "models/coercion_weights.json",
                                        "backend/ml/dataset_v2.py", "backend/ml/dataset.py", "backend/ml/features_v2.py",
                                        "backend/ml/replay_corpus.py", "data/coercion_corpus.jsonl", "scripts/train_model_v2.py",
                                        "backend/app/services/payee_check.py", "backend/app/services/evidence.py",
                                        "backend/app/services/graph_cache.py", "backend/app/services/payee_reputation.py"]
            if (ROOT / p).exists()}
    scripts = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "scripts/final_v2").glob("*.py"))}
    scripts.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "scripts/experiments").glob("*.py"))})
    outs = {str(p.relative_to(ROOT)): sha(p) for p in sorted(R.rglob("*")) if p.is_file() and p.suffix != ".svg"
            and "reproducibility" not in p.parts}
    head = (ROOT / "SNAPSHOT_GIT_HEAD").read_text().strip() if (ROOT / "SNAPSHOT_GIT_HEAD").exists() else "n/a"
    man = {"git_head_of_evaluated_snapshot": head,
           "environment_experiments": {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
                                       "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                                       "joblib": joblib.__version__, "matplotlib": matplotlib.__version__, "shap": shv},
           "environment_frozen_model": "scikit-learn 1.7.2 (the pickle loads under scikit-learn 1.7.2 with no InconsistentVersionWarning); the remaining build-environment versions are not recorded in models/metrics_v2.json",
           "environment_tests": "project machine: Python 3.10.12, pytest 9.1.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2",
           "seeds": {"frozen_v2_corpus_and_split": 42, "ten_seed_and_ablation": list(range(42, 52)), "replay_streams": [42, 43, 44],
                     "hgb_random_state": "= seed", "split_random_state": "= seed", "permutation_importance": 42, "shap_sample": 42},
           "dataset_configuration": {"n_transactions": 120000, "nominal_fraud_rate": 0.012, "n_payers": 2000, "dataset_version": frozen["dataset_version"],
                                     "fingerprint_seed42": frozen["dataset_fingerprint"]},
           "model_configuration": {"model": "HistGradientBoostingClassifier (max_iter 300, learning_rate 0.08, max_leaf_nodes 31, class_weight balanced)", "calibration": frozen["calibration"], "features": frozen["n_features"],
                                   "hgb": "max_iter 300, learning_rate 0.08, max_leaf_nodes 31, class_weight balanced"},
           "threshold_configuration": {"rule": "highest validation recall with validation FPR <= 1%, rounded to 4 d.p.",
                                       "frozen_threshold": frozen["validation_results"]["threshold"],
                                       "full_precision": frozen["validation_results"]["threshold_full_precision"]},
           "feature_configuration": {"features": frozen["features"], "groups": frozen["feature_groups"]},
           "runtime_substitutions_replay": ["backend.app.core.clock.now_app -> replay time t", "re._MAXCACHE = 8192 (performance)",
                                            "OMP_NUM_THREADS=1 (performance)"],
           "artefact_sha256": arts, "script_sha256": scripts, "output_sha256": outs}
    (R / "reproducibility").mkdir(exist_ok=True)
    (R / "reproducibility/reproducibility_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    cmds = """```bash
# from the repository root; JWT_SECRET must be set to any throwaway value
export OMP_NUM_THREADS=1                       # timing only
python scripts/final_v2/v2_classifier_analysis.py          # frozen V2, seed 42: tables 1-6, 12; figures 1-8, 11
python scripts/final_v2/v2_family_ablation.py --seeds 42 43 44 45 46 47 48 49 50 51   # 130 fits (~9 min on 2 vCPU)
python scripts/final_v2/v2_ablation_multiseed_summary.py   # tables 7, 8; figures 9, 10
python scripts/experiments/exp2b_template_disjoint.py --seeds 42 43 44 45 46 47 48 49 50 51   # template-disjoint control
for S in 42 43 44; do                                       # chronological replay (~95 min per stream)
  python scripts/experiments/exp3_4_replay.py --seed $S --n 60000 --db /tmp/replay_seed$S.db
  python scripts/experiments/exp4_rebuild_validation.py --seed $S --sample 60 --final-db /tmp/replay_seed$S.db
done
python scripts/final_v2/v2_end_to_end.py                    # ten-family re-aggregation with V2 in the Classifier family
python scripts/final_v2/v2_graph_and_replay_example.py      # graph statistics, before/assess/after example (reads the replay DBs)
python scripts/final_v2/v2_e2e_report.py                    # tables 9, 11; figures 12, 13, 15
python scripts/final_v2/v2_leakage_audit.py                 # LEAKAGE_AUDIT.md
python scripts/final_v2/v2_tests_graph_repro.py             # tables 10, 13; this manifest
python scripts/final_v2/polish_figures.py                   # final layout pass: figures 0, 2, 7-11 redrawn from saved results (no refit)
python scripts/final_v2/render_table_images.py              # table_images/*.png (needs node + playwright)
python scripts/final_v2/make_final_report.py                # FINAL_PROJECT_RESULTS.md
python -m pytest tests -q                                   # software tests (820)
```"""
    lines = ["# REPRODUCIBILITY MANIFEST — final V2 results", "",
             f"Evaluated code: snapshot of git HEAD `{head}` of the project repository (working tree with uncommitted changes; "
             "file hashes below identify the exact code and artefacts).", "",
             "## Environment", "",
             f"- Experiments: Python {man['environment_experiments']['python']} ({man['environment_experiments']['platform']}); numpy {numpy.__version__}, "
             f"pandas {pandas.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, joblib {joblib.__version__}, "
             f"matplotlib {matplotlib.__version__}, shap {shv}",
             f"- Frozen V2 model built with: {man['environment_frozen_model']}",
             f"- Software tests: {man['environment_tests']}", "",
             "## Seeds", "", *[f"- {k}: {v}" for k, v in man["seeds"].items()], "",
             "## Dataset configuration", "", *[f"- {k}: {v}" for k, v in man["dataset_configuration"].items() if k != "fingerprint_seed42"],
             f"- seed-42 dataset fingerprint (generator, corpus and coercion-weight hashes): see reproducibility_manifest.json", "",
             "## Model and threshold configuration", "", *[f"- {k}: {v}" for k, v in man["model_configuration"].items()],
             *[f"- threshold {k}: {v}" for k, v in man["threshold_configuration"].items()], "",
             "## Feature configuration", "", f"- 47 V2 features in 9 implementation groups: " +
             "; ".join(f"{g} ({len(c)})" for g, c in frozen["feature_groups"].items()), "",
             "## Runtime substitutions (replay only; none changes a prediction)", "", *[f"- {x}" for x in man["runtime_substitutions_replay"]], "",
             "## Key artefact hashes (SHA-256)", "", "| File | SHA-256 |", "|---|---|", *[f"| {k} | `{v}` |" for k, v in arts.items()], "",
             f"Hashes of all {len(scripts)} experiment scripts and {len(outs)} result files: `reproducibility_manifest.json`.", "",
             "## Commands", "", cmds, ""]
    (R / "reproducibility/REPRODUCIBILITY_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(t10.drop(columns="Modules list").to_string(index=False)); print(pd.DataFrame(kw).to_string(index=False))
    print(t13.T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
