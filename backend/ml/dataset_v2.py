"""Dataset v2: the v1 rows, unchanged, plus the channels v1 never carried.

The rule this module is built around
------------------------------------
v1 is not regenerated, reordered, relabelled or perturbed. generate_v2() calls
backend.ml.dataset.generate() and keeps its output exactly as it comes back -
same rows, same order, same labels, same twelve columns, bit for bit. Every v2
column is drawn afterwards from a SEPARATE random stream, so changing anything
about the v2 channels cannot move a v1 value. tests/test_dataset_v2.py asserts
all of that rather than trusting it.

Why more columns at all
-----------------------
The v1 audit measured where the limitation lives. Four model families trained
on the same twelve features land within 0.03 PR-AUC of one another on
validation, and social-engineering recall moves by at most 3.5 points across
them. What separates a scam payment from an ordinary one in the real world -
the message that prompted it, the link it carried, the payee's position in the
payment graph - is computed by the application and then never shown to the
model. v2 adds those channels to the feature vector. It does not make the
fraud easier to spot; several of the additions make it harder, deliberately.

What is NOT done here
---------------------
No fraud scenario is removed, no label is changed, no distribution of an
existing feature is touched, and no channel is drawn as a function of the
label that a rule could invert. The message channel in particular is built so
that "a message is present" carries almost no information (55% of legitimate
rows against 62% of social-engineering rows) and so that legitimate rows carry
the exact vocabulary a naive detector keys on - KYC, urgent, blocked, OTP,
verification - from the hard-negative half of data/coercion_corpus.jsonl.

Dataset version: "v2".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.ml import dataset as v1
from backend.ml.dataset import LABEL, generate as generate_v1

DATASET_VERSION = "v2"

# The v2 channels are drawn from their own generator, seeded apart from v1's.
# Nothing here may consume a draw from v1's stream, or the v1 columns would
# move when a v2 channel changed - which is the one thing this module promises
# cannot happen.
V2_SEED_OFFSET = 10_000

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "coercion_corpus.jsonl"

# Mirrors of the production gates, so a feature that the live graph refuses to
# compute is also missing in training. backend/app/services/graph_cache.py.
MIN_PAYERS_FOR_OVERLAP = 10
MIN_PAYERS_FOR_SHAPE = 5


# ── the message channel ─────────────────────────────────────────────────────
#
# P(a message accompanied this payment), by scenario. These are close together
# on purpose. If fraud always arrived with a message and legitimate payments
# never did, "message_available" would be the label and every metric below
# would be measuring the generator.
MESSAGE_PRESENT = {
    "legitimate": 0.55,
    "social_engineering": 0.62,
    "mule_collection": 0.45,
    "account_takeover": 0.25,      # the phone is stolen; nobody texts the victim
    "card_testing": 0.15,
}

# Which half of the corpus the message is drawn from, given that there is one.
# Legitimate payments carry hard negatives - real bank notices full of the
# words a naive detector treats as scam cues - and a few carry outright scam
# text, because receiving a scam message and then making an unrelated honest
# payment is an ordinary Tuesday. That last 4% is what stops "strong coercion
# pattern" from being a perfect rule.
MESSAGE_KIND = {
    "legitimate":         {"hard_negative": 0.46, "easy_negative": 0.50, "scam": 0.04},
    "social_engineering": {"scam": 0.65, "hard_negative": 0.25, "easy_negative": 0.10},
    "mule_collection":    {"scam": 0.35, "hard_negative": 0.35, "easy_negative": 0.30},
    "account_takeover":   {"hard_negative": 0.55, "easy_negative": 0.40, "scam": 0.05},
    "card_testing":       {"hard_negative": 0.50, "easy_negative": 0.45, "scam": 0.05},
}

# P(the message carried a link), and what kind of host it pointed at. The
# corpus itself contains no URLs, so the link channel is synthesised - and it
# is synthesised for legitimate messages too, at a similar rate, pointing at
# real billers and bank domains.
LINK_PRESENT = {"legitimate": 0.30, "social_engineering": 0.40, "mule_collection": 0.25,
                "account_takeover": 0.20, "card_testing": 0.20}

LEGITIMATE_HOSTS = (
    "https://www.onlinesbi.sbi/", "https://netbanking.hdfcbank.com/",
    "https://www.icicibank.com/", "https://www.npci.org.in/",
    "https://www.billdesk.com/pgmerc/", "https://www.irctc.co.in/",
    "https://www.amazon.in/orders", "https://pay.google.com/",
    "https://www.tneb.gov.in/bill", "https://www.myjio.com/recharge",
)
# Look-alikes: a real brand in a host that does not belong to it. These are the
# hosts backend/app/services/scam_link.py is built to catch.
IMPERSONATING_HOSTS = (
    "http://sbi-kyc-verify.xyz/update", "http://hdfcbank.secure-login.info/",
    "http://icicibank-refund.top/claim", "http://npci-upi-verify.online/",
    "http://paytm-cashback.site/win", "http://phonepe-support.click/helpdesk",
)
# Neither brand-claiming nor reassuring: shorteners and throwaway domains. Used
# by scams AND by ordinary small businesses, so the feature cannot separate on
# its own.
NEUTRAL_HOSTS = (
    "http://bit.ly/3xQ9k2", "http://tinyurl.com/y7m2b", "http://rb.gy/8qk1",
    "https://forms.gle/8Kd2n", "https://wa.me/message/7F2K", "https://linktr.ee/shopname",
)

# P(the payer stated a purpose) and the vocabulary the app offers.
INTENT_STATED = {"legitimate": 0.42, "social_engineering": 0.55, "mule_collection": 0.40,
                 "account_takeover": 0.20, "card_testing": 0.15}
INTENTS = ("bill", "shop", "friend", "refund", "investment", "asked_to")
# What the payer says they are doing, by scenario. A scam victim is often
# telling the truth as they understand it - "a refund", "I was asked to" - and
# plenty of honest payments are refunds too.
INTENT_MIX = {
    "legitimate":         {"bill": .30, "shop": .34, "friend": .22, "refund": .08,
                           "investment": .04, "asked_to": .02},
    "social_engineering": {"refund": .26, "asked_to": .24, "bill": .18, "investment": .16,
                           "shop": .10, "friend": .06},
    "mule_collection":    {"shop": .30, "asked_to": .20, "friend": .18, "investment": .14,
                           "refund": .10, "bill": .08},
    "account_takeover":   {"friend": .34, "shop": .26, "bill": .20, "refund": .10,
                           "asked_to": .06, "investment": .04},
    "card_testing":       {"shop": .46, "friend": .24, "bill": .16, "refund": .08,
                           "asked_to": .04, "investment": .02},
}


@dataclass(frozen=True)
class _Corpus:
    scam: tuple[str, ...]
    hard_negative: tuple[str, ...]
    easy_negative: tuple[str, ...]

    def draw(self, kind: str, rng: np.random.Generator) -> str:
        pool = getattr(self, kind)
        return str(pool[int(rng.integers(0, len(pool)))])


def load_corpus(path: Optional[Path] = None) -> _Corpus:
    """The message corpus the coercion model was fitted on.

    Built by scripts/build_coercion_corpus.py: 500 scams, 557 hard negatives
    (legitimate messages deliberately loaded with authority, urgency and fear
    vocabulary) and 101 easy negatives. Templated, not collected from real
    traffic - a limitation this project states rather than hides.
    """
    path = path or CORPUS
    if not path.exists():
        raise FileNotFoundError(
            f"No message corpus at {path}. Build it first:\n"
            "    python scripts/build_coercion_corpus.py"
        )
    buckets: dict[str, list[str]] = {"scam": [], "hard_negative": [], "easy_negative": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        buckets.setdefault(row["kind"], []).append(row["text"])
    return _Corpus(tuple(buckets["scam"]), tuple(buckets["hard_negative"]),
                   tuple(buckets["easy_negative"]))


# ── the graph channel ───────────────────────────────────────────────────────
#
# Drawn to be consistent with the payee columns v1 already produced, and drawn
# so that the legitimate shapes which previously caused false positives are
# present in quantity: popular merchants with large components and high
# overlap, new local shops with tiny payer pools, and households paying two
# young shops at once.
#
# The production gate is mirrored exactly: an overlap reading needs at least
# MIN_PAYERS_FOR_OVERLAP payers on BOTH sides, so below that the feature is
# missing here too, rather than zero.

def _graph_row(scenario: str, payers: float, age_days: float, repeat_ratio: float,
               observed: bool, rng: np.random.Generator) -> dict[str, float]:
    """Graph position for one payee, at the moment of this payment."""
    if not observed:
        # The deployment has no reputation row for this address, so it has no
        # graph position either: the same absence, reported the same way.
        return {"arrival_span_days": np.nan, "lifespan_days": np.nan,
                "payments_per_day": np.nan, "component_size": np.nan,
                "shared_payer_peak": np.nan, "overlap_ratio": np.nan,
                "graph_available": 0.0}

    payers = float(max(payers, 1.0))
    age = float(max(age_days, 0.5))

    if scenario == "mule_collection":
        # Payers arrive in a compressed window and almost none return.
        arrival_span = float(np.clip(rng.gamma(1.6, 1.8), 0.05, age))
        lifespan = float(np.clip(age * rng.uniform(0.25, 0.9), 0.5, age))
        rate = payers / max(lifespan, 0.5) * float(rng.uniform(0.8, 1.6))
        component = float(v1._count(rng.gamma(2.0, 9) + payers, 2, 400))
        peak_share = float(np.clip(rng.beta(4.0, 2.2), 0, 1))
    elif scenario == "social_engineering":
        # A collection account, unless it is one of the aged ones bought for
        # the purpose - those look like an ordinary small merchant.
        aged = age > 120
        arrival_span = float(np.clip(rng.gamma(2.6, 9 if aged else 2.2), 0.05, age))
        lifespan = float(np.clip(age * rng.uniform(0.4, 1.0), 0.5, age))
        rate = payers / max(lifespan, 0.5) * float(rng.uniform(0.5, 1.3))
        component = float(v1._count(rng.gamma(2.2, 12 if aged else 7) + payers, 2, 400))
        peak_share = float(np.clip(rng.beta(2.4 if aged else 3.2, 2.6), 0, 1))
    elif scenario == "account_takeover":
        arrival_span = float(np.clip(rng.gamma(2.2, 6), 0.05, age))
        lifespan = float(np.clip(age * rng.uniform(0.4, 1.0), 0.5, age))
        rate = payers / max(lifespan, 0.5) * float(rng.uniform(0.4, 1.1))
        component = float(v1._count(rng.gamma(2.4, 14) + payers, 2, 400))
        peak_share = float(np.clip(rng.beta(2.0, 3.0), 0, 1))
    elif scenario == "card_testing":
        arrival_span = float(np.clip(rng.gamma(1.4, 3), 0.05, age))
        lifespan = float(np.clip(age * rng.uniform(0.3, 1.0), 0.5, age))
        rate = payers / max(lifespan, 0.5) * float(rng.uniform(0.6, 1.5))
        component = float(v1._count(rng.gamma(2.0, 10) + payers, 2, 400))
        peak_share = float(np.clip(rng.beta(2.2, 3.0), 0, 1))
    else:
        # Legitimate, and deliberately varied. A popular merchant has hundreds
        # of payers, a large component and a high overlap with other merchants
        # its customers also use - the exact shape that was mistaken for a ring
        # before the overlap gate was added. A new local shop has almost none
        # of that. A household paying two young shops produces a high overlap
        # ratio over a tiny pool, which is why the gate is on payer COUNT.
        popular = payers >= 60 or repeat_ratio > 0.6
        young = age < 45
        arrival_span = float(np.clip(
            rng.gamma(3.0, 26 if popular else (2.6 if young else 12)), 0.05, age))
        lifespan = float(np.clip(age * rng.uniform(0.55, 1.0), 0.5, age))
        rate = payers / max(lifespan, 0.5) * float(rng.uniform(0.7, 2.4 if popular else 1.4))
        component = float(v1._count(
            rng.gamma(3.0, 40 if popular else (1.8 if young else 16)) + payers, 2, 900))
        peak_share = float(np.clip(rng.beta(3.0 if popular else 1.8, 2.4), 0, 1))

    shared_peak = float(v1._count(payers * peak_share, 0, 900))
    # Production computes an overlap ratio only when both payees clear the
    # fan-in gate. Below it the reading is refused, not set to zero.
    if payers >= MIN_PAYERS_FOR_OVERLAP and shared_peak >= 1:
        peer_payers = float(v1._count(
            max(shared_peak, rng.gamma(2.6, 14)), 1, 900))
        overlap = float(min(shared_peak / max(min(payers, peer_payers), 1.0), 1.0))
        if peer_payers < MIN_PAYERS_FOR_OVERLAP:
            overlap = np.nan
    else:
        overlap = np.nan

    return {
        "arrival_span_days": round(arrival_span, 3),
        "lifespan_days": round(lifespan, 3),
        "payments_per_day": round(float(np.clip(rate, 0.0, 400.0)), 3),
        "component_size": component,
        "shared_payer_peak": shared_peak,
        "overlap_ratio": overlap if np.isnan(overlap) else round(overlap, 4),
        "graph_available": 1.0,
    }


# ── the relationship channel ────────────────────────────────────────────────

def _relationship_row(payee_is_new: float, scenario: str,
                      rng: np.random.Generator) -> dict[str, float]:
    """How much this payer has paid THIS payee before, at decision time."""
    if payee_is_new >= 0.5:
        # Never paid before is a measured zero, not an absence: the payer's own
        # ledger is always readable, whatever is unknown about the payee.
        return {"prior_payments_to_payee": 0.0,
                "days_since_first_payment_to_payee": np.nan,
                "relationship_available": 1.0}
    prior = float(v1._count(rng.gamma(2.2, 3.0), 1, 300))
    days = float(np.clip(rng.gamma(2.6, 45), 1.0, 1200.0))
    return {"prior_payments_to_payee": prior,
            "days_since_first_payment_to_payee": round(days, 2),
            "relationship_available": 1.0}


# ── message, link and intent features, through the production extractors ────
#
# The features are computed by the same functions the running service calls -
# coercion.extract_features, coercion.probability, scam_link.analyse_links -
# so a value in the training matrix and a value at serving come from one
# implementation rather than two that agree today.

from backend.app.services import coercion as _coercion          # noqa: E402
from backend.app.services import scam_link as _scam_link        # noqa: E402

MESSAGE_PATTERN_FEATURES = tuple(f"msg_{code}" for code in _coercion.FEATURE_ORDER)
MESSAGE_FEATURES = MESSAGE_PATTERN_FEATURES + (
    "msg_coercion_probability", "msg_coercion_score", "msg_weak_cues_only",
    "msg_length_chars", "msg_word_count", "msg_digit_ratio",
    "msg_uppercase_ratio", "message_available",
)
LINK_FEATURES = ("link_present", "link_suspicious_score", "link_brand_conflict")
INTENT_FEATURES = ("intent_stated", "intent_is_refund_or_asked", "intent_payee_mismatch")
GRAPH_FEATURES = ("arrival_span_days", "lifespan_days", "payments_per_day",
                  "component_size", "shared_payer_peak", "overlap_ratio", "graph_available")
RELATIONSHIP_FEATURES = ("prior_payments_to_payee", "days_since_first_payment_to_payee",
                         "relationship_available")


@lru_cache(maxsize=65_536)
def _message_features_cached(text: str) -> tuple[tuple[str, float], ...]:
    analysis = _coercion.analyse_message(text)
    out = {f"msg_{code}": float(bool(value)) for code, value in analysis.patterns.items()}
    out["msg_coercion_probability"] = float(analysis.probability)
    out["msg_coercion_score"] = float(analysis.score) / 100.0
    out["msg_weak_cues_only"] = float(bool(analysis.weak_only))
    letters = [c for c in text if c.isalpha()]
    out["msg_length_chars"] = float(len(text))
    out["msg_word_count"] = float(len(text.split()))
    out["msg_digit_ratio"] = float(sum(c.isdigit() for c in text) / max(len(text), 1))
    out["msg_uppercase_ratio"] = float(sum(c.isupper() for c in letters) / max(len(letters), 1))
    out["message_available"] = 1.0
    return tuple(out.items())


@lru_cache(maxsize=65_536)
def _link_features_cached(text: str) -> tuple[tuple[str, Any], ...]:
    analysis = _scam_link.analyse_links(text)
    if not analysis.links:
        return (("link_present", 0.0), ("link_suspicious_score", 0.0),
                ("link_brand_conflict", 0.0), ("_claimed_brand", None))
    worst = max(analysis.links, key=lambda v: v.score)
    return (("link_present", 1.0),
            ("link_suspicious_score", float(min(worst.score, 100)) / 100.0),
            ("link_brand_conflict", 0.0),
            ("_claimed_brand", worst.claimed_brand))


def message_features(text: Optional[str]) -> dict[str, float]:
    """Every message feature, or the absence of all of them.

    No message is not a quiet message. When the payer did not supply one there
    is nothing to read, so the features are NaN and message_available says so -
    the same rule the payee columns already follow. Writing zeros here would
    tell the model "this text contained no pressure", which is a claim about a
    text that does not exist.
    """
    if not text:
        blank = {name: np.nan for name in MESSAGE_FEATURES}
        blank["message_available"] = 0.0
        return blank
    # analyse_message is the function the service itself calls, so the value
    # in this matrix and the value at serving come from one implementation.
    return dict(_message_features_cached(text))


def link_features(text: Optional[str], payee_brand: Optional[str]) -> dict[str, float]:
    """URL reputation, as the service computes it.

    A message with no URL is a measurement - we read the text and there was no
    link - so these are zeros, not NaN. A message that does not exist is an
    absence, and gets NaN.
    """
    if not text:
        return {name: np.nan for name in LINK_FEATURES}
    out = dict(_link_features_cached(text))
    claimed = out.pop("_claimed_brand", None)
    if claimed and payee_brand and str(claimed).lower() != str(payee_brand).lower():
        # The link speaks for one brand and the money goes somewhere else.
        # Honest messages do this too - a bank alert followed by paying a shop -
        # so it is a feature, not a rule.
        out["link_brand_conflict"] = 1.0
    return out


def intent_features(intent: Optional[str], payee_is_phone: float) -> dict[str, float]:
    """What the payer says this payment is for, and whether that fits the payee.

    A bill paid to a personal phone-number address is a mismatch worth noting;
    it is also something that happens honestly, which is why the generator
    produces it on both sides of the label.
    """
    if not intent:
        return {"intent_stated": 0.0, "intent_is_refund_or_asked": np.nan,
                "intent_payee_mismatch": np.nan}
    return {
        "intent_stated": 1.0,
        "intent_is_refund_or_asked": float(intent in ("refund", "asked_to", "investment")),
        "intent_payee_mismatch": float(intent in ("bill", "investment") and payee_is_phone >= 0.5),
    }


# ── the feature contract ────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "payer":        ["is_night", "hours_from_usual"],
    "amount":       ["amount", "amount_over_user_avg"],
    "velocity":     ["seconds_since_last_txn", "txns_last_hour", "txns_today_over_avg"],
    "payee":        ["payee_age_days", "payee_distinct_payers", "payee_repeat_ratio",
                     "payee_history_available"],
    "relationship": ["payee_is_new", *RELATIONSHIP_FEATURES],
    "graph":        list(GRAPH_FEATURES),
    "message":      list(MESSAGE_FEATURES),
    "link":         list(LINK_FEATURES),
    "intent":       list(INTENT_FEATURES),
}

FEATURES_V1: list[str] = list(v1.FEATURES)
FEATURES_V2: list[str] = FEATURES_V1 + [
    f for group in ("relationship", "graph", "message", "link", "intent")
    for f in FEATURE_GROUPS[group] if f not in FEATURES_V1
]

# Group membership for the ablation, over the v2 feature set.
def features_for(groups: list[str]) -> list[str]:
    """The feature columns belonging to a set of groups, in FEATURES_V2 order."""
    wanted = {f for g in groups for f in FEATURE_GROUPS[g]}
    missing = [g for g in groups if g not in FEATURE_GROUPS]
    if missing:
        raise KeyError(f"unknown feature group(s): {missing}")
    return [f for f in FEATURES_V2 if f in wanted]


def generate_v2(n_transactions: int = 120_000, fraud_rate: float = 0.012,
                n_users: int = 2_000, seed: int = 42,
                corpus: Optional[_Corpus] = None) -> pd.DataFrame:
    """v1's rows with the v2 channels appended. v1 columns are untouched."""
    df = generate_v1(n_transactions=n_transactions, fraud_rate=fraud_rate,
                     n_users=n_users, seed=seed)
    frozen = df[FEATURES_V1 + [LABEL, "scenario"]].copy(deep=True)

    rng = np.random.default_rng(seed + V2_SEED_OFFSET)
    corpus = corpus or load_corpus()

    scenarios = df["scenario"].to_numpy()
    payers = df["payee_distinct_payers"].to_numpy()
    ages = df["payee_age_days"].to_numpy()
    repeats = df["payee_repeat_ratio"].to_numpy()
    observed = df["payee_history_available"].to_numpy() >= 0.5
    is_new = df["payee_is_new"].to_numpy()

    rows: list[dict[str, float]] = []
    for i in range(len(df)):
        scen = str(scenarios[i])

        # Is this payee a personal phone-number address? Collection accounts
        # are more often one, and so are plenty of honest small sellers.
        p_phone = {"mule_collection": 0.55, "social_engineering": 0.48,
                   "card_testing": 0.40, "account_takeover": 0.35}.get(scen, 0.22)
        payee_is_phone = float(rng.random() < p_phone)
        payee_brand = None if payee_is_phone else str(
            rng.choice(["sbi", "hdfc", "icici", "paytm", "phonepe", "shop", "biller"]))

        # message
        text = None
        if rng.random() < MESSAGE_PRESENT.get(scen, 0.4):
            mix = MESSAGE_KIND[scen]
            kinds = list(mix); probs = np.array([mix[k] for k in kinds], dtype=float)
            kind = kinds[int(rng.choice(len(kinds), p=probs / probs.sum()))]
            text = corpus.draw(kind, rng)
            if rng.random() < LINK_PRESENT.get(scen, 0.25):
                if kind == "scam":
                    host = (IMPERSONATING_HOSTS if rng.random() < 0.55 else NEUTRAL_HOSTS)
                else:
                    # Legitimate messages carry real billers and bank domains,
                    # and sometimes a shortener, because small businesses do.
                    host = (LEGITIMATE_HOSTS if rng.random() < 0.82 else NEUTRAL_HOSTS)
                text = f"{text} {host[int(rng.integers(0, len(host)))]}"

        intent = None
        if rng.random() < INTENT_STATED.get(scen, 0.35):
            mix = INTENT_MIX[scen]
            names = list(mix); probs = np.array([mix[k] for k in names], dtype=float)
            intent = names[int(rng.choice(len(names), p=probs / probs.sum()))]

        row: dict[str, float] = {}
        row.update(_relationship_row(float(is_new[i]), scen, rng))
        row.update(_graph_row(scen, float(payers[i]) if not np.isnan(payers[i]) else 0.0,
                              float(ages[i]) if not np.isnan(ages[i]) else 0.0,
                              float(repeats[i]) if not np.isnan(repeats[i]) else 0.0,
                              bool(observed[i]), rng))
        row.update(message_features(text))
        row.update(link_features(text, payee_brand))
        row.update(intent_features(intent, payee_is_phone))
        row["payee_is_phone_address"] = payee_is_phone
        rows.append(row)

    extra = pd.DataFrame(rows, index=df.index)
    for column in extra.columns:
        df[column] = extra[column]
    df["dataset_version"] = DATASET_VERSION

    # The promise, enforced. If a v2 draw ever moved a v1 value, stop here.
    pd.testing.assert_frame_equal(
        df[FEATURES_V1 + [LABEL, "scenario"]], frozen, check_exact=True,
        obj="v1 columns after appending the v2 channels")
    return df


FEATURES_V2 = FEATURES_V2 + ["payee_is_phone_address"]
FEATURE_GROUPS["payee"] = FEATURE_GROUPS["payee"] + ["payee_is_phone_address"]


# ── a cache, with an integrity check ────────────────────────────────────────
#
# Generating 120,000 rows runs the coercion and link analysers 120,000 times,
# which takes a minute and a half. The cache is an optimisation and nothing
# more: it records the inputs that produced it - the seed, the size, and the
# hashes of this module and of the corpus - and refuses itself when any of them
# has moved. Reproducibility comes from the generator being deterministic; the
# cache just spares the clock.

import hashlib                                                   # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"


def _fingerprint(n: int, fraud_rate: float, n_users: int, seed: int) -> dict[str, Any]:
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "dataset_version": DATASET_VERSION, "rows": n, "fraud_rate": fraud_rate,
        "users": n_users, "seed": seed, "v2_seed_offset": V2_SEED_OFFSET,
        "dataset_v2_sha256": sha(Path(__file__)),
        "dataset_v1_sha256": sha(Path(v1.__file__)),
        "corpus_sha256": sha(CORPUS),
        "coercion_weights_sha256": sha(ROOT / "models" / "coercion_weights.json"),
        "features": FEATURES_V2,
    }


def load_or_generate_v2(n_transactions: int = 120_000, fraud_rate: float = 0.012,
                        n_users: int = 2_000, seed: int = 42,
                        use_cache: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The v2 dataset and the fingerprint of what produced it."""
    want = _fingerprint(n_transactions, fraud_rate, n_users, seed)
    stem = CACHE_DIR / f"dataset_v2_seed{seed}_n{n_transactions}"
    # joblib rather than parquet: it is already a dependency of this project,
    # and a cache is not worth a new one.
    frame, meta = stem.with_suffix(".joblib"), stem.with_suffix(".json")
    if use_cache and frame.exists() and meta.exists():
        have = json.loads(meta.read_text(encoding="utf-8"))
        if have == want:
            import joblib
            return joblib.load(frame), want
    df = generate_v2(n_transactions, fraud_rate, n_users, seed)
    if use_cache:
        import joblib
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(df, frame, compress=3)
        meta.write_text(json.dumps(want, indent=2), encoding="utf-8")
    return df, want
