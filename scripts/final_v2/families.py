"""Mapping of the 47 V2 features to the ten evidence families of the live system.

The V2 implementation groups its columns into nine FEATURE_GROUPS
(backend/ml/dataset_v2.py). The live assessment path is organised into ten
evidence families (backend/app/services/evidence.Family). This module maps each
V2 column to the live family that reads the same underlying observation:

  * amount_over_user_avg is payer-relative and feeds the payer-behaviour
    family's AMOUNT_DEVIATION fact, so it maps to Payer behaviour although the
    implementation lists it in the 'amount' group; the raw amount maps to
    Amount context (rail-limit / size checks);
  * payee_is_phone_address is a property of the address itself (address parsing),
    so it maps to Address and QR although the implementation lists it in 'payee';
  * payee_is_new (payer-relative novelty) is in the implementation's
    relationship group and maps to Relationship.

The tenth family, Classifier, is model-derived evidence: the V2 model itself.
It has no input column of its own; all 47 columns are its inputs.
"""

FAMILY_ORDER = ["Payer behaviour", "Amount context", "Payee history", "Relationship", "Graph",
                "Address and QR", "Message pressure", "Link safety", "Stated intent", "Classifier"]

FAMILY_OF = {
    # payer behaviour (implementation groups: payer, amount, velocity)
    "is_night": "Payer behaviour", "hours_from_usual": "Payer behaviour",
    "amount_over_user_avg": "Payer behaviour",
    "seconds_since_last_txn": "Payer behaviour", "txns_last_hour": "Payer behaviour",
    "txns_today_over_avg": "Payer behaviour",
    # amount context
    "amount": "Amount context",
    # payee history (implementation group: payee)
    "payee_age_days": "Payee history", "payee_distinct_payers": "Payee history",
    "payee_repeat_ratio": "Payee history", "payee_history_available": "Payee history",
    # address and QR (implementation group: payee)
    "payee_is_phone_address": "Address and QR",
    # relationship
    "payee_is_new": "Relationship", "prior_payments_to_payee": "Relationship",
    "days_since_first_payment_to_payee": "Relationship", "relationship_available": "Relationship",
    # graph
    "arrival_span_days": "Graph", "lifespan_days": "Graph", "payments_per_day": "Graph",
    "component_size": "Graph", "shared_payer_peak": "Graph", "overlap_ratio": "Graph",
    "graph_available": "Graph",
    # message pressure
    **{c: "Message pressure" for c in (
        "msg_authority", "msg_urgency", "msg_fear", "msg_pay_request", "msg_credential_request",
        "msg_secrecy", "msg_remote_control", "msg_verify_payment", "msg_lure", "msg_contact_offline",
        "msg_coercion_probability", "msg_coercion_score", "msg_weak_cues_only", "msg_length_chars",
        "msg_word_count", "msg_digit_ratio", "msg_uppercase_ratio", "message_available")},
    # link safety
    "link_present": "Link safety", "link_suspicious_score": "Link safety", "link_brand_conflict": "Link safety",
    # stated intent
    "intent_stated": "Stated intent", "intent_is_refund_or_asked": "Stated intent",
    "intent_payee_mismatch": "Stated intent",
}

assert len(FAMILY_OF) == 47
