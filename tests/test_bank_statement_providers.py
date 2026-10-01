"""Bank-statement provider adapters (SBI, Bank of Baroda).

All values are synthetic. The Bank of Baroda layout mirrors a real statement's
structure (centred rows, 43-char hard wrap, 49-char narration cap); the SBI
layout follows SBI's documented table and the supplied narration examples -
no real SBI PDF was available.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.app.ingestion.canonical import Direction
from backend.app.ingestion.identity import normalize_upi_id
from backend.app.ingestion.pipeline import import_statement
from backend.app.ingestion.providers import PdfDoc, detect
from backend.app.ingestion.providers.bank_of_baroda import BankOfBaroda
from backend.app.ingestion.providers.sbi import StateBankOfIndia
from backend.app.services import profile_store

from statement_pdf_fixtures import bob_like_pdf, sbi_like_pdf

BOB, SBI = BankOfBaroda(), StateBankOfIndia()

BOB_ROWS = [
    dict(date="01-02-2024", narration="UPI/400000000001/10:09:59/UPI/synthpersonname@axl", dr=120.0),
    dict(date="02-02-2024", narration="UPI/400000000002/17:13:29/UPI/9000000501@axl/Payment", dr=50.0),
    dict(date="03-02-2024", narration="UPI/400000000003/12:59:53/UPI/paytm-90000001@paytm", dr=30.0),
    dict(date="04-02-2024", narration="UPI/400000000004/08:00:00/UPI/9000000502@ybl/Paym", cr=500.0),
    dict(date="05-02-2024", narration="UPI/400000000005/09:00:00/UPI/paytmqr28100505010111abc", dr=10.0),
    dict(date="06-02-2024", narration="UPI/400000000006/09:30:00/UPI/synthlongname9@okicici", dr=15.0),
    dict(date="07-02-2024", narration="10000000000001:Int.Pd:01-11-2023 to 31-01-2024", cr=3.0),
    dict(date="08-02-2024", narration="UPI/400000000008/11:11:11/UPI/synth.shop@ybl/Paym", dr=75.0),
    dict(date="09-02-2024", narration="UPI/400000000009/12:12:12/UPI/synth.shop@ybl/Paym", dr=80.0),
]
SBI_ROWS = [
    dict(date="01/08/2024", narration="UPI/DR/627300000001/SYNTHPEE/BARB/synthpayee/Payment", dr=100.0),
    dict(date="02/08/2024", narration="UPI/DR/635300000002/Synth Ltd/UTIB/synthupi@/Payment", dr=250.0),
    dict(date="03/08/2024", narration="UPI/CR/103000000003/SYNTH SE/HDFC/a1@ybl/Request", cr=900.0),
    dict(date="04/08/2024", narration="UPI/DR/611600000004/Synth Ent/utib/gpay-12190/IT251", dr=40.0),
    dict(date="05/08/2024", narration="UPI/DR/611600000005/SYNTHNAMEWRAPPER/HDFC/b2@ybl/Pay", dr=60.0),
    dict(date="06/08/2024", narration="NEFT*SYNTH EMPLOYER*SALARY", cr=30000.0),
    dict(date="07/08/2024", narration="UPI/REV/611600000007", cr=60.0),
]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "bank.db")
    profile_store._SCHEMA_DONE.clear()
    yield tmp_path / "bank.db"
    profile_store._SCHEMA_DONE.clear()


def _bob():
    doc = PdfDoc.load(bob_like_pdf(BOB_ROWS))
    adapter, _ = detect(doc)
    return adapter, adapter.parse(doc)


def _sbi():
    doc = PdfDoc.load(sbi_like_pdf(SBI_ROWS))
    adapter, _ = detect(doc)
    return adapter, adapter.parse(doc)


# ── Identity normalisation ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["Muralinaga@BARB", "muralinaga@barb", '" muralinaga@barb "',
                                 "muralinaga @ barb", "UPI ID: muralinaga@barb",
                                 "​muralinaga@barb﻿"])
def test_one_normaliser_for_vpas(raw):
    assert normalize_upi_id(raw) == "muralinaga@barb"


@pytest.mark.parametrize("raw", ["Swiggy Ltd", "muralinaga", "swiggyupi@", "gpay-12190", "CHINTHAD"])
def test_merchant_strings_and_fragments_are_not_vpas(raw):
    assert normalize_upi_id(raw) is None


# ── Bank of Baroda grammar ───────────────────────────────────────────────────

@pytest.mark.parametrize("narration,upi,fragment", [
    ("UPI/439900000001/10:09:59/UPI/synthperson@axl", "synthperson@axl", None),          # @axl
    ("UPI/440700000002/17:13:29/UPI/9000000501@axl/Paym", "9000000501@axl", None),        # suffix
    ("UPI/404000000003/12:59:53/UPI/paytm-6700001@paytm", "paytm-6700001@paytm", None),   # @paytm
    ("UPI/404000000004/12:00:00/UPI/9000000502@ybl/Payment", "9000000502@ybl", None),     # @ybl
    ("UPI/404000000005/12:00:00/UPI/paytmqr28100505010111", None, "paytmqr28100505010111"),  # merchant id, no @
    ("UPI/404000000006/12:00:00/UPI/veerendragandham9@o", None, "veerendragandham9@o"),   # cut handle
])
def test_bob_narration_grammar(narration, upi, fragment):
    info = BOB.parse_narration(narration, at_cap=len(narration) >= 49)
    assert info["counterparty_upi_id"] == upi
    assert info["upi_fragment"] == fragment
    assert info["counterparty_name"] is None            # BoB prints no name
    assert info["reference_id"] == narration.split("/")[1]
    assert info["time"].strftime("%H:%M:%S") == narration.split("/")[2]


def test_bob_cut_handle_that_happens_to_validate_is_not_an_identity():
    n = "UPI/404000000007/12:00:00/UPI/synthlongname9@okic"      # from @okicici
    assert len(n) == 49
    info = BOB.parse_narration(n, at_cap=True)
    assert info["counterparty_upi_id"] is None and info["upi_fragment"] == "synthlongname9@okic"


def test_bob_display_label_is_marked_as_vpa_local_part_never_a_name():
    info = BOB.parse_narration("UPI/400000000001/10:09:59/UPI/synthperson@axl", at_cap=False)
    assert info["display_label"] == "synthperson"
    assert info["display_label_source"] == "VPA_LOCAL_PART"


def test_bob_pdf_rows_directions_wraps_and_skips():
    adapter, rows = _bob()
    assert adapter.name == "bank_of_baroda"
    assert len(rows) == len(BOB_ROWS)                     # opening balance / legend skipped
    first = rows[0]
    assert first.timestamp == datetime(2024, 2, 1, 10, 9, 59)
    assert first.timestamp_confidence == "HIGH"
    assert first.direction is Direction.DEBIT and first.amount == 120.0
    # wrapped across two lines in the PDF, reconstructed as one token
    assert first.counterparty_upi_id == "synthpersonname@axl"
    assert first.identity_state == "VPA_ONLY"
    assert rows[3].direction is Direction.CREDIT and rows[3].amount == 500.0
    assert rows[1].counterparty_upi_id == "9000000501@axl"
    interest = rows[6]
    assert interest.payment_method == "INTEREST" and interest.direction is Direction.CREDIT
    assert interest.identity_state == "UNKNOWN" and interest.timestamp_confidence == "MEDIUM"
    assert interest.time_known is False
    assert all(r.balance is not None for r in rows)


# ── SBI grammar ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("narration,name,bank,upi,fragment,direction", [
    ("UPI/DR/627300000001/SYNTH PE/BARB/synthpayee/Payme", "SYNTH PE", "BARB", None, "synthpayee", Direction.DEBIT),
    ("UPI/DR/635300000002/Synth Ltd/UTIB/synthupi@/Payment", "Synth Ltd", "UTIB", None, "synthupi@", Direction.DEBIT),
    ("UPI/DR/103000000003/SYNTH RAM/HDFC/synthramak/Reque", "SYNTH RAM", "HDFC", None, "synthramak", Direction.DEBIT),
    ("UPI/DR/611600000004/Synth Ent/UTIB/gpay-12190/IT251", "Synth Ent", "UTIB", None, "gpay-12190", Direction.DEBIT),
    ("UPI/CR/103000000005/SYNTH SE/HDFC/a1@ybl/Request", "SYNTH SE", "HDFC", "a1@ybl", None, Direction.CREDIT),
    ("UPI/DR/103000000006/SYNTH D/ICIC/98765@ybl/Pay", "SYNTH D", "ICIC", "98765@ybl", None, Direction.DEBIT),
])
def test_sbi_narration_grammar(narration, name, bank, upi, fragment, direction):
    info = SBI.parse_narration(narration, at_cap=False)
    assert info["counterparty_name"] == name
    assert info["counterparty_bank_code"] == bank
    assert info["counterparty_upi_id"] == upi
    assert info["upi_fragment"] == fragment
    assert info["narrated_direction"] is direction
    assert info["reference_id"] == narration.split("/")[2]


def test_sbi_no_vpa_narration_is_unknown_identity():
    info = SBI.parse_narration("NEFT*SYNTH EMPLOYER*SALARY", at_cap=False)
    assert info.get("counterparty_upi_id") is None and info.get("counterparty_name") is None


def test_sbi_pdf_rows_wrapped_name_and_vpa_and_direction():
    adapter, rows = _sbi()
    assert adapter.name == "sbi"
    assert len(rows) == len(SBI_ROWS)
    # wrapped mid-token / at a space / before a "/" -> one narration each
    assert rows[0].counterparty_name == "SYNTHPEE" and rows[0].upi_fragment == "synthpayee"
    assert rows[1].counterparty_name == "Synth Ltd" and rows[1].upi_fragment == "synthupi@"
    assert rows[3].counterparty_bank_code == "UTIB"
    assert rows[6].transaction_type == "REVERSAL" and rows[6].identity_state == "UNKNOWN"
    assert rows[0].identity_state == "NAME_ONLY"
    assert rows[0].direction is Direction.DEBIT
    assert rows[2].direction is Direction.CREDIT and rows[2].counterparty_upi_id == "a1@ybl"
    assert rows[2].identity_state == "EXACT_VPA"
    # name wrapped across the line break
    assert rows[4].counterparty_name == "SYNTHNAMEWRAPPER" and rows[4].counterparty_upi_id == "b2@ybl"
    assert rows[0].timestamp == datetime(2024, 8, 1) and rows[0].timestamp_confidence == "MEDIUM"
    assert all("**" not in (r.narration or "") for r in rows)          # footer not absorbed
    assert rows[5].identity_state == "UNKNOWN" and rows[5].payment_method == "NEFT"


def test_sbi_narration_contradicting_the_columns_leaves_direction_unknown():
    rows = [dict(date="01/08/2024", narration="UPI/CR/627300000001/SYNTH A/HDFC/c3@ybl/Pay", dr=10.0)] * 3
    doc = PdfDoc.load(sbi_like_pdf(rows))
    adapter, _ = detect(doc)
    parsed = adapter.parse(doc)
    assert all(r.direction is None and "direction_conflict" in r.reasons for r in parsed)


# ── Detection ────────────────────────────────────────────────────────────────

def test_bank_names_in_free_text_do_not_select_a_parser(tmp_path):
    from statement_pdf_fixtures import bank_like_table_pdf

    rows = [("01/07/2026", "UPI to synth@okaxis via sbi axis paytm tez", "1", "10.00", "", "90.00")]
    adapter, detections = detect(PdfDoc.load(bank_like_table_pdf(rows * 3)))
    assert adapter is None
    assert all(d.confidence < 0.6 for d in detections)


def test_detection_reports_confidence_and_evidence():
    _, detections = detect(PdfDoc.load(bob_like_pdf(BOB_ROWS)))
    best = detections[0]
    assert best.provider == "bank_of_baroda" and best.confidence >= 0.6
    assert any("grammar" in e for e in best.evidence)


# ── Through the canonical pipeline into payee history and /payee/check ────────

def test_bob_import_counts_and_idempotence(db):
    r = import_statement("u1", "bob.pdf", bob_like_pdf(BOB_ROWS),
                         imported_at=datetime(2024, 3, 1))
    rep = r.report
    assert rep["source_provider"] == "bank_of_baroda"
    assert rep["total_rows_detected"] == len(BOB_ROWS)
    assert (rep["debit_transactions"], rep["credit_transactions"]) == (7, 2)
    assert rep["counterparty_names_found"] == 0
    # the 49-char cap cuts "…@paytm" to "…@payt" and "…@okicici" to "…@okic";
    # "paytmqr…" prints no handle at all
    assert rep["partial_identifiers_found"] == 2
    assert rep["non_vpa_identifiers_found"] == 1
    assert rep["counterparty_vpas_found"] == rep["complete_vpas_found"] == 5
    assert rep["upi_transactions"] == 8
    again = import_statement("u1", "bob.pdf", bob_like_pdf(BOB_ROWS), imported_at=datetime(2024, 3, 1))
    assert again.report["stored_transactions"] == 0
    assert again.report["payee_relationships_updated"] == 0


def test_no_timestamp_is_invented(db):
    import sqlite3

    import_statement("u1", "bob.pdf", bob_like_pdf(BOB_ROWS), imported_at=datetime(2024, 3, 1))
    conn = sqlite3.connect(db)
    stamps = [r[0] for r in conn.execute("SELECT timestamp FROM statement_transactions")]
    conn.close()
    assert all(s.startswith("2024-02-") for s in stamps)


def _api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "api.db")
    profile_store._SCHEMA_DONE.clear()
    import backend.main as main

    client = TestClient(main.app)
    token = client.post("/auth/register", json={"username": "synth.payer@okaxis",
                                                "password": "hunter2hunter2"}).json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.mark.parametrize("pdf_kind", ["bob", "sbi"])
def test_statement_vpa_is_recognised_as_paid_before_at_check_time(tmp_path, monkeypatch, pdf_kind):
    import backend.app.services.payee_check as pc

    captured = {}
    real = pc.build_vector
    monkeypatch.setattr(pc, "build_vector",
                        lambda **kw: captured.setdefault("v", real(**kw)) and real(**kw))
    api = _api(tmp_path, monkeypatch)
    content = bob_like_pdf(BOB_ROWS) if pdf_kind == "bob" else sbi_like_pdf(SBI_ROWS)
    up = api.post("/statement/upload", files={"file": ("statement.pdf", content, "application/pdf")})
    assert up.status_code == 200, up.text
    assert "extracted_transactions" not in up.json()
    # the same address, typed differently
    target = " SYNTH.SHOP@YBL " if pdf_kind == "bob" else "B2 @ ybl"
    expected = "PAID_BEFORE" if pdf_kind == "bob" else "PAID_ONCE"
    res = api.post("/payee/check", json={"payload": target, "amount": 100}).json()
    codes = {f["code"] for f in res["findings"]}
    assert res["payee_status"] == expected, res["payee_status"]
    assert res["history_state"]["state"] == "HISTORY_FOUND"
    assert res["payee_history"]["sources"] == {"STATEMENT_OBSERVED": res["payee_history"]["payment_count"]}
    assert "payee_unseen" not in codes
    assert next(e for e in res["evidence"] if e["family"] == "relationship")["code"] == "relationship_established"
    assert "No payment history for this address" not in str(res)
    if res["ml"]["available"]:
        # V1 read the statement relationship (unavailable only when the payer
        # has too few payments for an amount baseline).
        assert captured["v"].values["payee_is_new"] == 0.0
        assert captured["v"].values["payee_history_available"] == 1.0
    else:
        assert pdf_kind == "sbi"


def test_a_different_vpa_stays_unknown(tmp_path, monkeypatch):
    api = _api(tmp_path, monkeypatch)
    api.post("/statement/upload", files={"file": ("s.pdf", bob_like_pdf(BOB_ROWS), "application/pdf")})
    res = api.post("/payee/check", json={"payload": "synth.shop2@ybl", "amount": 100}).json()
    assert res["payee_status"] == "UNKNOWN"
    assert res["history_state"]["state"] == "NO_HISTORY"


def test_a_truncated_vpa_never_matches_its_completion(tmp_path, monkeypatch):
    """The statement shows 'synthlongname9@okic'. The QR's real address
    'synthlongname9@okicici' must not be treated as paid before."""
    api = _api(tmp_path, monkeypatch)
    api.post("/statement/upload", files={"file": ("s.pdf", bob_like_pdf(BOB_ROWS), "application/pdf")})
    res = api.post("/payee/check", json={"payload": "synthlongname9@okicici", "amount": 100}).json()
    assert res["payee_history"]["payment_count"] == 0


def test_sbi_name_only_rows_are_never_identity(tmp_path, monkeypatch):
    """SBI prints 'SYNTHPEE' and the fragment 'synthpayee'. A QR for
    synthpayee@okaxis named 'SYNTHPEE' is explained, not matched."""
    api = _api(tmp_path, monkeypatch)
    api.post("/statement/upload", files={"file": ("s.pdf", sbi_like_pdf(SBI_ROWS), "application/pdf")})
    res = api.post("/payee/check", json={
        "payload": "upi://pay?pa=synthpayee@okaxis&pn=SYNTHPEE", "amount": 100}).json()
    assert res["payee_history"]["payment_count"] == 0
    assert res["payee_status"] in ("PARTIAL_IDENTITY", "NAME_ONLY_HISTORY")


# ── Identifier types and partial history (bank-statement task 2) ──────────────

@pytest.mark.parametrize("narration,itype,identifier,upi", [
    ("UPI/DR/627300000001/SYNTHNAM/BARB/synthpayee/Payme", "PARTIAL_VPA_FRAGMENT", "synthpayee", None),
    ("UPI/DR/635300000002/Synth Ltd/UTIB/synthupi@/Payment", "PARTIAL_VPA_FRAGMENT", "synthupi@", None),
    ("UPI/DR/611600000004/Synth Ent/UTIB/gpay-12190/IT251", "PARTIAL_VPA_FRAGMENT", "gpay-12190", None),
    ("UPI/CR/103000000005/SYNTH SE/HDFC/a1@ybl/Request", "COMPLETE_VPA", "a1@ybl", "a1@ybl"),
    ("UPI/DR/103000000006/SYNTHNAM/BARB//Payme", "NAME_ONLY", None, None),
])
def test_sbi_identifier_types_never_invent_a_handle(narration, itype, identifier, upi):
    info = SBI.parse_narration(narration, at_cap=False)
    assert info["identifier_type"] == itype
    assert info["counterparty_identifier"] == identifier
    assert info["counterparty_upi_id"] == upi
    assert info["name_source"] == "BANK_NARRATION"
    # the bank code is never turned into a handle
    assert not (info["counterparty_upi_id"] or "").endswith(("@barb", "@utib"))


def test_sbi_reversal_is_a_reversal_not_a_payment():
    info = SBI.parse_narration("UPI/REV/611600000007", at_cap=False)
    assert info["transaction_type"] == "REVERSAL" and info["reference_id"] == "611600000007"


@pytest.mark.parametrize("narration,itype,identifier", [
    ("UPI/439900000001/10:09:59/UPI/synthperson@axl", "COMPLETE_VPA", "synthperson@axl"),
    ("UPI/440700000002/16:16:52/UPI/79900000001@ybl/Pay", "COMPLETE_VPA", "79900000001@ybl"),
    ("UPI/404100000003/21:13:35/UPI/synthupi@axisbank/", "COMPLETE_VPA", "synthupi@axisbank"),
    ("UPI/404000000004/12:59:53/UPI/paytm-6700001@paytm", "COMPLETE_VPA", "paytm-6700001@paytm"),
    ("UPI/404500000005/06:02:51/UPI/synthname2211", "NON_VPA_IDENTIFIER", "synthname2211"),
    ("UPI/403200000006/12:20:47/UPI/paytmqr281005050101", "NON_VPA_IDENTIFIER", "paytmqr281005050101"),
    ("UPI/404000000007/12:00:00/UPI/veerendragandham3@o", "PARTIAL_VPA_FRAGMENT", "veerendragandham3@o"),
    ("UPI/404000000008/12:00:00/UPI/bhimrewards.m2p@hdf", "PARTIAL_VPA_FRAGMENT", "bhimrewards.m2p@hdf"),
])
def test_bob_identifier_types_keep_the_whole_identifier(narration, itype, identifier):
    info = BOB.parse_narration(narration, at_cap=len(narration) >= 49)
    assert info["identifier_type"] == itype
    assert info["counterparty_identifier"] == identifier     # never cut before "@"
    assert (info["counterparty_upi_id"] is not None) == (itype == "COMPLETE_VPA")


def test_sbi_wrap_rules_hyphen_and_at(tmp_path):
    rows = [dict(date="01/08/2024", narration="UPI/DR/611600000011/Lala Ent/UTIB/gpay-12190/IT251", dr=5.0),
            dict(date="02/08/2024", narration="UPI/DR/611600000012/SYNTHNAM/HDFC/abc@okaxis/Pay", dr=6.0),
            dict(date="03/08/2024", narration="UPI/DR/611600000013/SAGI RAM/HDFC/sagiramakr/Reque", dr=7.0)]
    doc = PdfDoc.load(sbi_like_pdf(rows, wrap_width=150.0))
    adapter, _ = detect(doc)
    parsed = adapter.parse(doc)
    assert [r.counterparty_identifier for r in parsed] == ["gpay-12190", "abc@okaxis", "sagiramakr"]
    assert parsed[2].counterparty_name == "SAGI RAM"
    assert all(r.transaction_id for r in parsed)


def test_sbi_partial_identifier_is_explained_not_matched(tmp_path, monkeypatch):
    api = _api(tmp_path, monkeypatch)
    api.post("/statement/upload", files={"file": ("s.pdf", sbi_like_pdf(SBI_ROWS), "application/pdf")})
    res = api.post("/payee/check", json={"payload": "synthpayee@barb", "amount": 100}).json()
    assert res["payee_status"] == "PARTIAL_IDENTITY"
    assert res["history_state"]["state"] == "PARTIAL_IDENTITY"
    assert "synthpayee" in res["history_state"]["message"]
    assert "cannot be verified" in res["history_state"]["message"]
    assert res["payee_history"]["payment_count"] == 0            # never "paid before"
    rel = next(e for e in res["evidence"] if e["family"] == "relationship")
    assert rel["code"] == "relationship_unconfirmed"
    # a fragment is never matched to an address it does not begin
    other = api.post("/payee/check", json={"payload": "someoneelse@barb", "amount": 100}).json()
    assert other["payee_status"] == "UNKNOWN"


def test_reference_id_dedupe_keeps_same_date_same_amount_payments(db):
    rows = [dict(date="01/08/2024", narration=f"UPI/DR/61160000002{i}/SYNTHNAM/HDFC/synthpay{i}x/Pay", dr=50.0)
            for i in range(3)]
    r = import_statement("u1", "s.pdf", sbi_like_pdf(rows), imported_at=datetime(2024, 9, 1))
    assert r.report["stored_transactions"] == 3
    again = import_statement("u1", "s.pdf", sbi_like_pdf(rows), imported_at=datetime(2024, 9, 1))
    assert again.report["stored_transactions"] == 0 and again.report["duplicate_transactions"] == 3


def test_partial_identifiers_never_create_payee_relationships(db):
    r = import_statement("u1", "s.pdf", sbi_like_pdf(SBI_ROWS), imported_at=datetime(2024, 9, 1))
    rep = r.report
    # only the one COMPLETE VPA debit (b2@ybl) becomes an exact relationship
    assert rep["payee_relationships_updated"] == 1
    assert rep["partial_identifiers_found"] == 3        # synthpayee, synthupi@, gpay-12190
    assert rep["reversal_transactions"] == 1
    assert rep["counterparty_names_found"] == 5
