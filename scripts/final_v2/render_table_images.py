"""Render the main V2 result tables as PNG images (HTML + headless Chromium) (results/final_v2/table_images/).

Every cell is read from results/final_v2/tables/*.csv or the leakage-audit JSON;
nothing is recomputed. Wide tables are rendered with a documented column subset;
the full tables remain in tables/*.md and *.csv.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

R = ROOT / "results" / "final_v2"
T = R / "tables"
OUT = R / "table_images"


CSS = """
body{margin:0;background:#ffffff;font-family:'DejaVu Sans',Arial,sans-serif;color:#0b0b0b}
.wrap{display:inline-block;padding:18px 22px 16px 22px;background:#ffffff}
h1{font-size:15px;font-weight:600;margin:0 0 10px 0}
table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{background:#e8eef8;font-weight:600;text-align:right;padding:6px 10px;border-bottom:1.5px solid #9aa7ba;vertical-align:bottom;max-width:150px}
td{padding:4px 10px;text-align:right;border-bottom:1px solid #e4e3df;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}
tr:nth-child(even) td{background:#f7f6f3}
tr.hl td{font-weight:700}
.note{font-size:11px;color:#52514e;margin-top:9px;max-width:%dpx;line-height:1.35}
"""


def render(df: pd.DataFrame, name: str, title: str, note: str = "", highlight=None, left_cols=()):
    import html
    df = df.astype(str).replace({"nan": "—", "": "—"})
    cols = list(df.columns)
    import re
    num = re.compile(r"^[\s0-9.,+\-−±%/()—≤≥=x×]*$")
    auto = {c for c in cols if (df[c].map(lambda v: not num.match(v)).mean() > 0.5) and c not in ("Status",)}
    lc = set(left_cols) | {cols[0]} | auto
    th = "".join(f'<th class="{"l" if c in lc else ""}">{html.escape(c)}</th>' for c in cols)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        cls = ' class="hl"' if highlight and highlight(i, r) else ""
        rows.append(f"<tr{cls}>" + "".join(f'<td class="{"l" if c in lc else ""}">{html.escape(str(r[c]))}</td>' for c in cols) + "</tr>")
    body = (f'<div class="wrap" id="t"><h1>{html.escape(title)}</h1><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            + (f'<div class="note">{html.escape(note)}</div>' if note else "") + "</div>")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / f"{name}.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS % 900}</style>{body}", encoding="utf-8")
    JOBS.append(name)


BUILD = Path("/tmp/claude-0/table_html")
JOBS: list[str] = []


def shoot():
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    js = BUILD / "shoot.js"
    js.write_text("""
const { chromium } = require('playwright');
(async () => {
  const [build, out, ...names] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ deviceScaleFactor: 3, viewport: { width: 2400, height: 1200 } });
  for (const n of names) {
    await p.goto('file://' + build + '/' + n + '.html');
    await (await p.$('#t')).screenshot({ path: out + '/' + n + '.png' });
  }
  await b.close();
})();""")
    subprocess.run(["node", str(js), str(BUILD), str(OUT), *JOBS], check=True,
                   env={**__import__("os").environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()})


def csv(name):
    return pd.read_csv(T / f"{name}.csv", dtype=str, keep_default_na=False)


def main() -> int:
    render(csv("table01_dataset_splits"), "table01_dataset_splits", "Table 1 — Dataset and split statistics (V2 corpus, seed 42)",
           "120,000 synthetic payments, 2,000 simulated payers; stratified split 72,000 / 18,000 / 30,000.")
    render(csv("table02_scenarios"), "table02_scenarios", "Table 2 — Fraud scenario composition",
           "Generator shares are simulator settings (backend/ml/dataset.SCENARIO_MIX), not estimates of real UPI fraud.")
    t3 = csv("table03_feature_inventory")
    keep = [c for c in t3.columns if c in ("#", "Feature", "Evidence family", "Type", "Missing possible", "Missing % (seed 42)")]
    render(t3[keep], "table03_feature_inventory", "Table 3 — V2 feature inventory (47 features)",
           "Column subset; full meanings in tables/table03_feature_inventory.md. All features are computed from state before the payment.",
           left_cols=("Feature", "Evidence family", "Type"))
    render(csv("table04_overall_performance"), "table04_overall_performance",
           "Table 4 — Overall V2 classification performance (frozen model, seed 42, threshold 0.1123)",
           "Threshold selected on the validation split only; the test split was scored once.")
    render(csv("table04c_classification_report"), "table04c_classification_report", "Table 4c — Classification report (V2, test split)")
    render(csv("table05_confusion_matrix"), "table05_confusion_matrix", "Table 5 — V2 test confusion matrix (threshold 0.1123)",
           "Precision 0.4843 · Recall 0.7803 · Specificity 0.9900 · FPR 0.00995 · F1 0.5976 · MCC 0.6091")
    render(csv("table06_scenario_performance"), "table06_scenario_performance", "Table 6 — Scenario-level detection (V2, test split)",
           "Per-scenario precision is not defined: false positives are legitimate payments and belong to no scenario.")
    t7 = csv("table07_v2_family_ablation")
    keep7 = ["Configuration", "F", "PR-AUC", "ΔPR-AUC (pp)", "ROC-AUC", "Recall %", "FPR %", "SE recall %", "MC recall %",
             "AT recall %", "CT recall %", "Seeds lower PR-AUC"]
    render(t7[keep7], "table07_v2_family_ablation", "Table 7 — V2 feature-family ablation (mean over seeds 42–51)",
           "Each configuration refitted with the identical protocol; Δ paired by seed (configuration − full V2), mean ± sd. "
           "SE/MC/AT/CT = synthetic scenario recall. Associations under the simulator, not causal effects.",
           highlight=lambda i, row: i == 0)
    render(csv("table08_ten_seed_stability"), "table08_ten_seed_stability", "Table 8 — V2 ten-seed stability (seeds 42–51)",
           "Each seed regenerates the corpus, refits V2, selects its threshold on its own validation split and measures its test split once. sd with ddof = 1.")
    t9 = csv("table09_end_to_end")
    render(t9, "table09_end_to_end", "Table 9 — Ten-family end-to-end evaluation (chronological replay, V2 as the Classifier family)",
           "Verdicts are advisory bands of one risk score, not ground-truth classes. Three replay streams (seeds 42–44), 60,000 payments each.")
    render(csv("table09b_verdict_distribution"), "table09b_verdict_distribution", "Table 9b — Verdict distribution by group (last 25% of each stream)",
           "Rates are mean ± sd over three streams; counts per stream (seed 42 / 43 / 44).")
    render(csv("table09f_family_behaviour"), "table09f_family_behaviour", "Table 9f — Evidence-family behaviour in the complete system (last 25%)")
    t10 = csv("table10_software_tests").drop(columns=["Modules list"], errors="ignore")
    render(t10, "table10_software_tests", "Table 10 — Software test validation (pytest, project machine)",
           "Implementation-correctness results, not fraud-detection accuracy. Per-module list in tables/table10_software_tests.md.",
           highlight=lambda i, row: str(row.iloc[0]).startswith("Total"))
    render(csv("table11_replay_validation"), "table11_replay_validation", "Table 11 — Chronological graph-replay validation (three streams)")
    render(csv("table12a_top_features"), "table12a_top_features", "Table 12a — Top 15 V2 features by permutation importance (test split)")
    render(csv("table12b_family_importance"), "table12b_family_importance", "Table 12b — Evidence-family importance (V2, test split)")
    t13 = csv("table13_graph_statistics")
    keep13 = [c for c in t13.columns if c in ("Stream", "Payer nodes", "Payee nodes", "Payer–payee edges", "Repeat edges (≥2 payments)",
                                              "Connected components", "Largest component (nodes)", "Components = independent rebuild",
                                              "Graph evidence available at assessment (%)", "Positional finding fired (fraud / legit)")]
    render(t13[keep13], "table13_graph_statistics", "Table 13 — Replayed payer–payee graph (end of each 150-day stream)",
           "Explicit incremental graph; no graph neural network and no embeddings.")
    la = json.load(open(R / "leakage_audit/leakage_audit_v2.json"))
    tl = pd.DataFrame([{"#": x["#"], "Check": x["check"], "Status": x["status"]} for x in la["items"]])
    render(tl, "table14_leakage_audit", "Leakage audit — 16 PASS · 0 FAIL · 2 NOT APPLICABLE",
           "Evidence for every item: leakage_audit/LEAKAGE_AUDIT.md.")
    for old in OUT.glob("*.svg"):
        old.unlink()
    shoot()
    print(sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
