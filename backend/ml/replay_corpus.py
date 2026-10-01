"""A chronological, identity-bearing payment stream for replay experiments.

Why a separate corpus
---------------------
backend/ml/dataset.py and dataset_v2.py produce ROW-LEVEL data: every row is
the evidence state at the moment one payment is about to be made, sampled
from scenario-conditional distributions. There are no payer or payee
identities and no clock, so the running system (ledger, reputation
aggregates, incremental payer-payee graph, payer profiles) cannot be replayed
over them. This module generates a separate stream that CAN be replayed:

  * persistent payer and payee identities (VPAs, QR payloads);
  * monotonic timestamps (naive IST, as backend/app/core/clock.py stores them);
  * scenario-controlled fraud, with the scenario mixture of dataset.py;
  * payer behaviour drawn from the SAME user model (dataset._make_users) and
    amount/hour distributions that mirror dataset.py's scenario generators;
  * message / link / stated-purpose channels drawn with the SAME
    scenario-conditional probabilities and the SAME templated message corpus
    as dataset_v2.generate_v2.

It does NOT modify, import-mutate or re-seed the V1/V2 generators; the
row-level predictive corpus is unchanged. Deterministic given `seed`.

What the stream contains is a modelling choice, stated in REPLAY_DESIGN and
recorded in every experiment manifest. In particular the collection-account
behaviour (how many victims, how compressed, whether victims pay several
accounts of one ring) is designed here; the graph evidence the running system
derives from it is a consequence of that design, not an independent fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

from backend.ml import dataset as v1
from backend.ml.dataset_v2 import (
    IMPERSONATING_HOSTS, INTENT_MIX, INTENT_STATED, LEGITIMATE_HOSTS, LINK_PRESENT,
    MESSAGE_KIND, MESSAGE_PRESENT, NEUTRAL_HOSTS, load_corpus,
)

START = datetime(2026, 1, 1, 0, 0, 0)
UPI_CAP = v1.UPI_PER_TXN_CAP

REPLAY_DESIGN = {
    "window_days": 150,
    "n_payers": 2000,
    "n_merchants": 900,
    "late_opening_merchant_share": 0.30,
    "merchant_popularity": "Zipf-like weights w_i = 1 / (rank_i ** 1.05)",
    "qr_share_for_merchant_payments": 0.5,
    "fraud_rate_target": 0.012,
    "scenario_mix": dict(v1.SCENARIO_MIX),
    "social_engineering": {
        "victims_per_collector": "Poisson(7)+2", "active_window_days": "gamma(2, 6)",
        "aged_collector_share": 0.35,
        "aged_collector": "an existing low-popularity merchant address (bought/stolen)",
        "brand_lure_handle_share_of_fresh": 0.30,
        "qr_with_other_display_name_share": 0.20,
    },
    "mule_collection": {
        "accounts_per_ring": "3-4", "victims_per_ring": "20-45",
        "ring_window_days": "gamma(1.6, 2.5) clipped to [1, 20]",
        "victim_pays_two_ring_accounts": 0.6,
    },
    "account_takeover": {
        "burst_payments": "1-3 within minutes", "to_previously_paid_payee": 0.25,
        "hours": "as dataset._account_takeover",
    },
    "card_testing": {"probes_per_event": "3-8 within minutes",
                     "amount": "uniform(1, 25)", "testing_payees_shared": True},
    "observation": "every executed payment is confirmed and recorded AFTER its assessment",
}

_FIRST = ("ravi", "anita", "suresh", "lakshmi", "arjun", "priya", "kiran", "deepa", "rahul",
          "sneha", "vijay", "kavya", "manoj", "divya", "sanjay", "pooja", "ajay", "meena",
          "naveen", "swathi", "gopal", "rekha", "harish", "bhavana", "srinivas", "latha")
_LAST = ("kumar", "reddy", "rao", "sharma", "naidu", "varma", "iyer", "das", "patel",
         "singh", "raju", "prasad", "nair", "gupta", "murthy", "chowdary")
_SHOP_A = ("ganesh", "sri", "lakshmi", "balaji", "sai", "venkata", "annapurna", "durga",
           "krishna", "anand", "royal", "new", "city", "green", "star", "metro", "super",
           "raja", "hari", "om", "vasavi", "sree", "maruthi", "surya")
_SHOP_B = ("stores", "medicals", "textiles", "bakery", "tiffins", "mart", "electricals",
           "traders", "hardware", "mobiles", "fruits", "sweets", "provisions", "opticals",
           "tailors", "fancy", "books", "pharmacy", "restaurant", "enterprises")
_HANDLES_BANK = ("okaxis", "okhdfcbank", "okicici", "oksbi")
_HANDLES_APP = ("ybl", "ibl", "axl", "paytm")
_BRAND_LURE = ("sbi.kyc", "hdfc.refund", "icici.update", "npci.support", "paytm.cashback",
               "phonepe.help", "customer.care", "refund.desk", "kyc.verify")


@dataclass
class Payee:
    key: str
    kind: str                  # merchant | person | se_fresh | se_aged | mule | ato | tester
    display: Optional[str] = None
    popularity: float = 0.0
    opens_day: float = 0.0
    is_phone: bool = False


@dataclass
class Payer:
    idx: int
    user: Any                  # dataset._User
    phone_vpa: str
    habitual: list = field(default_factory=list)   # payee keys, the payer's own circle
    landlord: Optional[str] = None


def _phone(rng) -> str:
    return str(int(rng.integers(6_000_000_000, 9_999_999_999)))


def _hour_normal(rng, mean, sd) -> int:
    return int(np.clip(rng.normal(mean, sd), 0, 23))


def _moment(day: float, hour: int, rng) -> datetime:
    d = int(day)
    return START + timedelta(days=d, hours=hour, minutes=int(rng.integers(0, 60)),
                             seconds=int(rng.integers(0, 60)))


def _channels(rng, scenario: str, corpus) -> dict[str, Any]:
    """Message, link and stated purpose - the dataset_v2 probabilities."""
    text = None
    kind = None
    if rng.random() < MESSAGE_PRESENT.get(scenario, 0.4):
        mix = MESSAGE_KIND[scenario]
        kinds = list(mix); probs = np.array([mix[k] for k in kinds], dtype=float)
        kind = kinds[int(rng.choice(len(kinds), p=probs / probs.sum()))]
        text = corpus.draw(kind, rng)
        if rng.random() < LINK_PRESENT.get(scenario, 0.25):
            if kind == "scam":
                host = IMPERSONATING_HOSTS if rng.random() < 0.55 else NEUTRAL_HOSTS
            else:
                host = LEGITIMATE_HOSTS if rng.random() < 0.82 else NEUTRAL_HOSTS
            text = f"{text} {host[int(rng.integers(0, len(host)))]}"
    intent = None
    if rng.random() < INTENT_STATED.get(scenario, 0.35):
        mix = INTENT_MIX[scenario]
        names = list(mix); probs = np.array([mix[k] for k in names], dtype=float)
        intent = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    return {"message": text, "message_kind": kind, "intent": intent}


def generate_replay(seed: int = 42, n_transactions: int = 60_000,
                    fraud_rate: float = 0.012, n_payers: int = 2000,
                    window_days: int = 150, n_merchants: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corpus = load_corpus()
    users = v1._make_users(rng, n_payers)

    # ── payees ──────────────────────────────────────────────────────────────
    payees: dict[str, Payee] = {}
    merchants: list[str] = []
    for i in range(n_merchants):
        a = _SHOP_A[int(rng.integers(0, len(_SHOP_A)))]
        b = _SHOP_B[int(rng.integers(0, len(_SHOP_B)))]
        handle = (_HANDLES_BANK + _HANDLES_APP)[int(rng.integers(0, 8))]
        key = f"{a}{b}{i}@{handle}"
        display = f"{a.title()} {b.title()}"
        opens = float(rng.uniform(0, window_days * 0.8)) if rng.random() < 0.30 else 0.0
        payees[key] = Payee(key, "merchant", display,
                            popularity=1.0 / ((i + 1) ** 1.05), opens_day=opens)
        merchants.append(key)
    pop = np.array([payees[m].popularity for m in merchants]); pop = pop / pop.sum()

    def new_person() -> str:
        if rng.random() < 0.5:
            key = f"{_phone(rng)}@{_HANDLES_APP[int(rng.integers(0, 4))]}"
            phone = True
        else:
            key = (f"{_FIRST[int(rng.integers(0, len(_FIRST)))]}."
                   f"{_LAST[int(rng.integers(0, len(_LAST)))]}{int(rng.integers(10, 999))}"
                   f"@{(_HANDLES_BANK + _HANDLES_APP)[int(rng.integers(0, 8))]}")
            phone = False
        if key not in payees:
            payees[key] = Payee(key, "person", None, is_phone=phone)
        return key

    payers: list[Payer] = []
    for i, u in enumerate(users):
        p = Payer(i, u, phone_vpa=f"{_phone(rng)}@{_HANDLES_APP[int(rng.integers(0, 4))]}")
        k = int(np.clip(u.known_payees // 4, 3, 15))
        circle = list(rng.choice(merchants, size=k, replace=False, p=pop))
        circle += [new_person() for _ in range(max(1, k // 3))]
        p.habitual = circle
        if u.pays_large:
            p.landlord = new_person()
        payers.append(p)
    rates = np.array([p.user.daily_txns for p in payers]); rates = rates / rates.sum()

    rows: list[dict[str, Any]] = []
    eid = [0]

    def emit(payer: Payer, key: str, when: datetime, amount: float, scenario: str,
             event: int, qr: bool, qr_name: Optional[str] = None, group: str = "",
             channels: Optional[dict] = None):
        pe = payees[key]
        if qr:
            name = qr_name if qr_name is not None else (pe.display or "")
            payload = f"upi://pay?pa={key}" + (f"&pn={quote(name)}" if name else "")
        else:
            payload = key
        ch = channels if channels is not None else _channels(rng, scenario, corpus)
        rows.append({
            "timestamp": when, "payer_id": f"payer{payer.idx:04d}", "payee_key": key,
            "payee_kind": pe.kind, "payload": payload, "amount": round(float(amount), 2),
            "is_fraud": int(scenario != "legitimate"), "scenario": scenario,
            "event_id": event, "group_id": group, "payee_is_phone": bool(pe.is_phone),
            "payee_display": pe.display, **ch,
        })

    def next_event() -> int:
        eid[0] += 1
        return eid[0]

    # ── fraud budget by scenario (transactions, not events) ─────────────────
    n_fraud = int(round(n_transactions * fraud_rate))
    budget = {s: int(round(n_fraud * w)) for s, w in v1.SCENARIO_MIX.items()}
    n_legit = n_transactions - sum(budget.values())

    # ── legitimate stream ───────────────────────────────────────────────────
    produced = 0
    while produced < n_legit:
        payer = payers[int(rng.choice(len(payers), p=rates))]
        u = payer.user
        day = float(rng.uniform(0, window_days))
        kind = rng.random()
        high = 0.14 if u.pays_large else 0.012
        burst = 1
        if kind < high:
            ceiling = UPI_CAP if u.pays_large else float(np.clip(u.typical_amount * 60, 22_000, UPI_CAP))
            floor = 20_000.0 if u.pays_large else 15_000.0
            amount = v1._amount(np.exp(rng.uniform(np.log(floor), np.log(max(ceiling, floor * 1.2)))))
            new_payee = rng.random() < 0.5
            hour = _hour_normal(rng, u.usual_hour, 5.0)
            target = payer.landlord if (u.pays_large and payer.landlord and not new_payee) else None
        elif kind < high + 0.10:
            amount = v1._amount(u.typical_amount * float(np.clip(rng.gamma(3.0, 3.2), 1.5, 90)))
            new_payee = rng.random() < 0.55
            hour = _hour_normal(rng, u.usual_hour, 5.5); target = None
        elif kind < high + 0.16:
            amount = None; new_payee = rng.random() < 0.30
            hour = _hour_normal(rng, u.usual_hour, 3.0); target = None
            burst = int(rng.integers(2, 4))
        else:
            amount = v1._amount(np.exp(rng.normal(np.log(u.typical_amount), 0.85)))
            new_payee = rng.random() < 0.18
            hour = _hour_normal(rng, u.usual_hour, 3.6); target = None
        when = _moment(day, hour, rng)
        event = next_event()
        for b in range(min(burst, n_legit - produced)):
            if target is None:
                if new_payee:
                    if rng.random() < 0.7:
                        open_now = [m for m in rng.choice(merchants, size=8, replace=True, p=pop)
                                    if payees[m].opens_day <= day]
                        key = open_now[0] if open_now else new_person()
                    else:
                        key = new_person()
                    if key not in payer.habitual and rng.random() < 0.4:
                        payer.habitual.append(key)
                else:
                    key = payer.habitual[int(rng.integers(0, len(payer.habitual)))]
                    if payees[key].kind == "merchant" and payees[key].opens_day > day:
                        key = new_person()
            else:
                key = target
            amt = amount if amount is not None else v1._amount(
                np.exp(rng.normal(np.log(u.typical_amount) - 0.6, 0.7)))
            qr = payees[key].kind == "merchant" and rng.random() < 0.5
            emit(payer, key, when, amt, "legitimate", event, qr)
            when = when + timedelta(seconds=float(np.clip(rng.exponential(150), 20, 1500)))
            produced += 1

    # ── social engineering: collectors, some aged ───────────────────────────
    made = 0; c = 0
    while made < budget["social_engineering"]:
        c += 1
        aged = rng.random() < 0.35
        if aged:
            tail = merchants[len(merchants) // 2:]
            key = tail[int(rng.integers(0, len(tail)))]
            coll_kind = "se_aged"
        else:
            if rng.random() < 0.30:
                key = f"{_BRAND_LURE[int(rng.integers(0, len(_BRAND_LURE)))]}{int(rng.integers(10, 999))}@{_HANDLES_APP[int(rng.integers(0, 4))]}"
                payees[key] = Payee(key, "se_fresh", None)
            else:
                key = new_person(); payees[key].kind = "se_fresh"
            coll_kind = "se_fresh"
        start = float(rng.uniform(5, window_days - 3))
        span = float(np.clip(rng.gamma(2.0, 6.0), 1, 40))
        for _ in range(int(rng.poisson(7)) + 2):
            if made >= budget["social_engineering"]:
                break
            victim = payers[int(rng.integers(0, len(payers)))]
            u = victim.user
            roll = rng.random()
            if roll < 0.5:
                amount = v1._amount(u.typical_amount * float(np.clip(rng.gamma(1.6, 1.1), 0.2, 12)))
            elif roll < 0.85:
                amount = v1._amount(u.typical_amount * float(np.clip(rng.gamma(4.0, 2.6), 0.8, 80)))
            else:
                amount = v1._amount(np.exp(rng.uniform(np.log(25_000), np.log(UPI_CAP))))
            day = min(start + float(rng.uniform(0, span)), window_days - 0.01)
            when = _moment(day, _hour_normal(rng, u.usual_hour, 3.6), rng)
            qr = rng.random() < 0.20
            qr_name = ("SBI Customer Care", "HDFC Refund Desk", "Amazon Support",
                       "Electricity Board")[int(rng.integers(0, 4))] if qr else None
            emit(victim, key, when, amount, "social_engineering", next_event(), qr, qr_name,
                 group=f"se{c}:{coll_kind}")
            made += 1

    # ── mule collection: rings sharing victims ──────────────────────────────
    made = 0; r = 0
    while made < budget["mule_collection"]:
        r += 1
        ring = []
        for _ in range(int(rng.integers(3, 5))):
            key = new_person(); payees[key].kind = "mule"; ring.append(key)
        start = float(rng.uniform(3, window_days - 3))
        span = float(np.clip(rng.gamma(1.6, 2.5), 1, 20))
        for _ in range(int(rng.integers(20, 46))):
            if made >= budget["mule_collection"]:
                break
            victim = payers[int(rng.integers(0, len(payers)))]
            u = victim.user
            day = min(start + float(rng.uniform(0, span)), window_days - 0.01)
            when = _moment(day, _hour_normal(rng, u.usual_hour, 4.5), rng)
            targets = list(rng.choice(ring, size=2, replace=False)) if rng.random() < 0.6 else [ring[int(rng.integers(0, len(ring)))]]
            ev = next_event()
            for key in targets:
                if made >= budget["mule_collection"]:
                    break
                amount = v1._amount(u.typical_amount * float(np.clip(rng.gamma(3.0, 1.9), 0.5, 40)))
                emit(victim, key, when, amount, "mule_collection", ev, False, group=f"ring{r}")
                when = when + timedelta(minutes=float(rng.uniform(2, 180)))
                made += 1

    # ── account takeover: bursts from the victim's own account ─────────────
    made = 0
    while made < budget["account_takeover"]:
        victim = payers[int(rng.integers(0, len(payers)))]
        u = victim.user
        day = float(rng.uniform(5, window_days - 0.5))
        hour = int(rng.choice([1, 2, 3, 4, 23, 0, 14, 15], p=[.16, .16, .16, .12, .12, .12, .08, .08]))
        when = _moment(day, hour, rng)
        known = rng.random() < 0.25
        key = victim.habitual[int(rng.integers(0, len(victim.habitual)))] if known else new_person()
        if not known:
            payees[key].kind = "ato"
        ev = next_event()
        for _ in range(int(rng.integers(1, 4))):
            if made >= budget["account_takeover"]:
                break
            if rng.random() < 0.34:
                amount = v1._amount(UPI_CAP * float(rng.uniform(0.72, 1.0)))
            else:
                amount = v1._amount(u.typical_amount * float(np.clip(rng.gamma(3.0, 3.5), 0.6, 60)))
            emit(victim, key, when, amount, "account_takeover", ev, False, group=f"ato{ev}")
            when = when + timedelta(seconds=float(np.clip(rng.exponential(120), 3, 900)))
            made += 1

    # ── card testing: tiny probes in quick succession ──────────────────────
    made = 0
    testers: list[str] = []
    while made < budget["card_testing"]:
        if not testers or rng.random() < 0.35:
            key = f"{_SHOP_A[int(rng.integers(0, len(_SHOP_A)))]}{_SHOP_B[int(rng.integers(0, len(_SHOP_B)))]}x{int(rng.integers(100, 999))}@{_HANDLES_APP[int(rng.integers(0, 4))]}"
            payees[key] = Payee(key, "tester", None); testers.append(key)
        key = testers[int(rng.integers(0, len(testers)))]
        victim = payers[int(rng.integers(0, len(payers)))]
        when = _moment(float(rng.uniform(3, window_days - 0.5)), int(rng.integers(0, 24)), rng)
        ev = next_event()
        for _ in range(int(rng.integers(3, 9))):
            if made >= budget["card_testing"]:
                break
            emit(victim, key, when, float(rng.uniform(1, 25)), "card_testing", ev, False,
                 group=f"ct{ev}")
            when = when + timedelta(seconds=float(np.clip(rng.exponential(35), 1, 300)))
            made += 1

    df = pd.DataFrame(rows)
    df = df[df["timestamp"] < START + timedelta(days=window_days + 1)]
    # Chronological order; ties broken by a stable key so the order is a
    # deterministic function of the seed.
    df = df.sort_values(["timestamp", "event_id", "payer_id", "payee_key"], kind="mergesort")
    df = df.reset_index(drop=True)
    # Strictly increasing assessment times: a same-second tie would make
    # "strictly before t" ambiguous between two payments. Nudge ties by 1 ms,
    # preserving order.
    ts = df["timestamp"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1
    df["timestamp"] = pd.to_datetime(ts, unit="ms")
    df.insert(0, "txn_id", [f"t{i:06d}" for i in range(len(df))])
    df["replay_seed"] = seed
    return df
