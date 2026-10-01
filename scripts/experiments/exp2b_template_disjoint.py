"""Leakage control for the message channel: template-disjoint evaluation.

Finding of the leakage audit (results/audit/leakage_audit.json, item
"message_template_overlap"): the V2 message channel draws each message from a
fixed corpus of 1,022 templates (data/coercion_corpus.jsonl), independently
for every row. Train, validation and test rows therefore share the SAME
template texts, and the coercion weights that turn a text into features were
fitted on that same corpus. Test performance of the message family then
measures recognition of seen templates as well as generalisation.

This experiment removes the overlap and re-measures, for the same ten seeds:

  * the 1,022 templates are split 50/50 within each kind (scam / hard negative
    / easy negative) into partition A and partition B (rng seed 10_000+seed);
  * the coercion model is REFITTED on partition A only, with the recipe of
    backend/ml/coercion_model.py (LogisticRegression(class_weight='balanced'),
    threshold = lowest with <=5% hard-negative FPR measured on the fit set);
  * train and validation rows carry only partition-A texts, test rows only
    partition-B texts: a row whose drawn template is in the wrong partition
    gets a template of the SAME KIND from the right partition (rng seed
    20_000+seed); the appended link host, the label, the scenario and every
    non-message column are unchanged;
  * message and link features are recomputed with the production extractors;
  * configurations E (core+message), I (core+rel+graph+msg) and K (full V2) are
    refitted with the identical protocol (common.fit_select_evaluate).

Nothing else changes, so the paired difference (disjoint minus standard, same
seed) isolates the effect of template overlap on the reported numbers.

    python scripts/experiments/exp2b_template_disjoint.py --seeds 42 43 ... 51
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (RESULTS, environment, fit_select_evaluate, source_state,  # noqa: E402
                    stratified_splits, write_json)

import backend.ml.dataset_v2 as d2                                       # noqa: E402
from backend.app.services import coercion as C                           # noqa: E402
from backend.ml.coercion_model import choose_threshold, gated_scores, to_matrix  # noqa: E402

OUT = RESULTS / "ablation"
KINDS = ("scam", "hard_negative", "easy_negative")
MSG_COLS = list(d2.MESSAGE_FEATURES); LINK_COLS = list(d2.LINK_FEATURES)
G = d2.FEATURE_GROUPS
CORE = G["payer"] + G["amount"] + G["velocity"] + G["payee"]
CONFIGS = {
    "E_core+message": CORE + G["message"],
    "I_core+rel+graph+msg": CORE + G["relationship"] + G["graph"] + G["message"],
    "K_full_v2": list(d2.FEATURES_V2),
}


def generate_recording(seed: int):
    """generate_v2 with the per-row message text and payee brand recorded."""
    texts, brands = [], []
    orig_msg, orig_link = d2.message_features, d2.link_features

    def rec_msg(text):
        texts.append(text)
        return orig_msg(text)

    def rec_link(text, brand):
        brands.append(brand)
        return orig_link(text, brand)
    d2.message_features, d2.link_features = rec_msg, rec_link
    try:
        df = d2.generate_v2(120_000, 0.012, 2_000, seed)
    finally:
        d2.message_features, d2.link_features = orig_msg, orig_link
    assert len(texts) == len(df) == len(brands)
    return df, texts, brands


def fit_coercion(templates: list[str], labels: np.ndarray, kinds: list[str], seed: int) -> dict:
    X = to_matrix(templates)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(X, labels)
    probs = gated_scores(model.predict_proba(X)[:, 1], X)
    return {"feature_order": list(C.FEATURE_ORDER),
            "coefficients": {c: float(v) for c, v in zip(C.FEATURE_ORDER, model.coef_[0])},
            "intercept": float(model.intercept_[0]),
            "threshold": choose_threshold(probs, kinds)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = ap.parse_args()
    corpus = d2.load_corpus()
    kind_of = {t: k for k in KINDS for t in getattr(corpus, k)}
    rows, meta = [], {}
    std = pd.read_csv(OUT / "ablation_results.csv")

    for seed in args.seeds:
        t0 = time.time()
        C.reset_weights_cache(); d2._message_features_cached.cache_clear(); d2._link_features_cached.cache_clear()
        df, texts, brands = generate_recording(seed)
        cached, _ = d2.load_or_generate_v2(seed=seed)
        pd.testing.assert_frame_equal(df, cached, check_exact=True)   # identical to the Exp 1/2 data
        y = df["is_fraud"].to_numpy(); scen = df["scenario"].to_numpy()
        tr, va, te = stratified_splits(y, seed)
        split = np.empty(len(df), dtype="<U1"); split[tr] = "A"; split[va] = "A"; split[te] = "B"

        rng = np.random.default_rng(10_000 + seed)
        part = {}
        for k in KINDS:
            pool = list(getattr(corpus, k)); order = rng.permutation(len(pool))
            half = len(pool) // 2
            part[k] = {"A": [pool[j] for j in order[:half]], "B": [pool[j] for j in order[half:]]}
        in_part = {t: p for k in KINDS for p in ("A", "B") for t in part[k][p]}

        # decompose each recorded text into (template, suffix)
        base = []
        for t in texts:
            if not t:
                base.append(None); continue
            if t in kind_of:
                base.append((t, "")); continue
            head, tail = t.rsplit(" ", 1)
            assert head in kind_of, "unrecognised message text"
            base.append((head, " " + tail))

        def overlap(bases, idx_train, idx_test) -> float:
            seen = {bases[i][0] for i in idx_train if bases[i]}
            test = [bases[i][0] for i in idx_test if bases[i]]
            return float(np.mean([t in seen for t in test])) if test else float("nan")

        std_overlap = overlap(base, tr, te)
        rng2 = np.random.default_rng(20_000 + seed)
        new_base, n_swapped = [], 0
        for i, b in enumerate(base):
            if b is None:
                new_base.append(None); continue
            tmpl, suf = b
            want = split[i]
            if in_part[tmpl] != want:
                pool = part[kind_of[tmpl]][want]
                tmpl = pool[int(rng2.integers(0, len(pool)))]; n_swapped += 1
            new_base.append((tmpl, suf))
        dis_overlap = overlap(new_base, tr, te)

        # coercion model refitted on partition A only
        a_t = [t for k in KINDS for t in part[k]["A"]]
        a_k = [kind_of[t] for t in a_t]
        a_y = np.array([int(k == "scam") for k in a_k])
        w = fit_coercion(a_t, a_y, a_k, seed)
        C._weights = w
        d2._message_features_cached.cache_clear(); d2._link_features_cached.cache_clear()

        dis = df.copy()
        mrows = [d2.message_features(None if b is None else b[0] + b[1]) for b in new_base]
        lrows = [d2.link_features(None if b is None else b[0] + b[1], br) for b, br in zip(new_base, brands)]
        dis[MSG_COLS] = pd.DataFrame(mrows, index=df.index)[MSG_COLS]
        dis[LINK_COLS] = pd.DataFrame(lrows, index=df.index)[LINK_COLS]
        C.reset_weights_cache(); d2._message_features_cached.cache_clear(); d2._link_features_cached.cache_clear()
        assert (dis.drop(columns=MSG_COLS + LINK_COLS).equals(df.drop(columns=MSG_COLS + LINK_COLS)))

        meta[seed] = {"test_template_seen_in_train_standard": std_overlap,
                      "test_template_seen_in_train_disjoint": dis_overlap,
                      "rows_with_message": int(sum(b is not None for b in base)),
                      "rows_template_swapped": n_swapped,
                      "coercion_refit_on_A": w, "partition_sizes": {k: {p: len(v) for p, v in part[k].items()} for k in KINDS}}
        for name, cols in CONFIGS.items():
            r = fit_select_evaluate(dis.iloc[tr][cols], y[tr], dis.iloc[va][cols], y[va], dis.iloc[te][cols],
                                    y[te], scen[te], seed, scen_va=scen[va])
            r.pop("_model"); r.pop("_test_probs")
            s = std[(std.seed == seed) & (std.config == name)].iloc[0]
            r.update({"seed": seed, "config": name, "variant": "template_disjoint", "n_features": len(cols),
                      "standard_pr_auc": float(s.pr_auc), "standard_recall": float(s.recall),
                      "standard_recall_social_engineering": float(s.recall_social_engineering)})
            rows.append(r)
            print(f"seed {seed} {name:<22} PR-AUC {r['pr_auc']:.4f} (std {s.pr_auc:.4f})  rec {r['recall']:.4f} "
                  f"FPR {r['fpr']:.4f} SE {r['recall_social_engineering']:.3f} (std {s.recall_social_engineering:.3f})",
                  flush=True)
        print(f"seed {seed}: overlap std {std_overlap:.3f} -> disjoint {dis_overlap:.3f}; "
              f"swapped {n_swapped}; {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "template_disjoint_results.csv", index=False)

    res = pd.DataFrame(rows)
    summary = {}
    for name in CONFIGS:
        sub = res[res.config == name]
        summary[name] = {}
        for m, sm in (("pr_auc", "standard_pr_auc"), ("recall", "standard_recall"),
                      ("recall_social_engineering", "standard_recall_social_engineering")):
            d = (sub[m] - sub[sm]).to_numpy()
            summary[name][m] = {"disjoint_mean": float(sub[m].mean()), "disjoint_std": float(sub[m].std(ddof=1)),
                                "standard_mean": float(sub[sm].mean()), "standard_std": float(sub[sm].std(ddof=1)),
                                "paired_delta_pp_mean": float(100 * d.mean()),
                                "paired_delta_pp_std": float(100 * d.std(ddof=1)),
                                "seeds_lower_when_disjoint": int((d < 0).sum())}
        for m in ("roc_auc", "fpr", "precision", "f1", "brier"):
            summary[name][m] = {"disjoint_mean": float(sub[m].mean()), "disjoint_std": float(sub[m].std(ddof=1))}
    write_json(OUT / "template_disjoint_summary.json", {
        "what": "Message-template leakage control: train/val and test rows use disjoint template halves and "
                "the coercion weights are refitted on the training half only.",
        "seeds": args.seeds, "summary": summary, "per_seed": meta,
        "environment": environment(), "source": source_state()})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
