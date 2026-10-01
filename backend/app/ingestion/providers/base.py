"""Shared machinery for bank-statement PDF adapters.

Bank PDFs are tables drawn as text. pdfplumber gives words with coordinates;
this module turns them into rows WITHOUT parsing text lines independently:

1. The header row (found by each bank's own column names) fixes the column
   x-ranges: date, narration, and the debit / credit / balance amounts.
2. Every date in the date column is a row ANCHOR.
3. Narration words are assigned to an anchor by position. Two layouts exist:
   *centred* (the narration's first line sits ABOVE the date, a continuation
   BELOW it - Bank of Baroda) and *top-aligned* (date on the first line,
   continuation lines below - most others). The layout is detected from the
   page, not assumed.
4. Narration lines of one row are joined. A line that filled the column's
   fixed width is a hard wrap ("chittibabudar" + "am@axl" -> one token), so
   it is joined without a space; a shorter line ends at a real word boundary.
5. Direction comes from WHICH amount column holds the number - never from
   the narration and never by default.

Nothing here knows any bank's narration grammar; adapters own that.
"""

from __future__ import annotations

import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from backend.app.ingestion.canonical import CanonicalTransaction, Direction
from backend.app.ingestion.identity import normalize_upi_id


@dataclass
class Detection:
    provider: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    page: int


@dataclass
class PdfDoc:
    pages: list[list[Word]]
    text: str

    @classmethod
    def load(cls, content: bytes) -> "PdfDoc":
        import pdfplumber

        pages: list[list[Word]] = []
        texts: list[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
                pages.append([Word(w["text"], w["x0"], w["x1"], w["top"], i) for w in words])
                texts.append(page.extract_text() or "")
        return cls(pages=pages, text="\n".join(texts))


@dataclass
class Columns:
    """x-ranges of the statement table, from the header row."""
    header_top: float
    header_bottom: float
    date_x1: float                 # a date word starts left of this
    narration_x0: float
    narration_x1: float
    debit_x1: float                # right edge of the debit/withdrawal header
    credit_x1: float
    balance_x1: float


@dataclass
class RawRow:
    page: int
    date_text: str
    narration_lines: list[str]
    line_right_edges: list[float]
    debit: Optional[float]
    credit: Optional[float]
    balance: Optional[float]
    amount_ambiguous: bool = False


_AMOUNT = re.compile(r"^(-?[\d,]+\.\d{2})(?:\s*(?:Cr|Dr|CR|DR)\.?)?$")


def parse_amount(text: str) -> Optional[float]:
    """'1,234.50' -> 1234.5. A balance's Cr/Dr marker may be glued on
    ('500.00Cr'); it says which side of zero the BALANCE is, never the
    transaction's direction, which comes from the column."""
    m = _AMOUNT.match(text.strip())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


OPTIONAL_ROLES = {"ref"}      # e.g. SBI's statement has no cheque/ref column


def find_columns(words: list[Word], spec: dict[str, tuple[str, ...]],
                 narration_from: str = "after_date") -> Optional[Columns]:
    """Locate the header by the adapter's column names.

    `spec` maps date / narration / debit / credit / balance to header words
    (upper-cased first tokens). All five must sit on one line.
    """
    by_line: dict[int, list[Word]] = {}
    for w in words:
        by_line.setdefault(round(w.top), []).append(w)
    for top in sorted(by_line):
        line = [w for t in (top - 1, top, top + 1) for w in by_line.get(t, [])]
        found: dict[str, Word] = {}
        for role, names in spec.items():
            for w in line:
                if w.text.upper().strip(":") in names:
                    found.setdefault(role, w)
        required = {role for role in spec if role not in OPTIONAL_ROLES}
        if required <= set(found):
            def right_edge(role: str) -> float:
                # "(DR)" / "(CR)" / "Amount" may follow the header word.
                anchor = found[role]
                tail = [w for w in line if 0 <= w.x0 - anchor.x1 <= 25]
                return max([anchor.x1] + [w.x1 for w in tail])
            return Columns(
                header_top=min(w.top for w in found.values()),
                header_bottom=max(w.top for w in found.values()) + 6,
                # The date column ends a little right of its header; the
                # narration column starts there (its text begins left of the
                # centred "NARRATION" header).
                date_x1=found["date"].x1 + 8,
                narration_x0=(found["date"].x1 + 8 if narration_from == "after_date"
                              else found["narration"].x0 - 3),
                narration_x1=min(found["debit"].x0, found.get("ref", found["debit"]).x0) - 2,
                debit_x1=right_edge("debit"),
                credit_x1=right_edge("credit"),
                balance_x1=right_edge("balance"),
            )
    return None


def _date_tokens(words: list[Word], x_limit: float, pattern: re.Pattern) -> list[Word]:
    """Dates in the date column, including multi-word ones ("1 Aug 2024"):
    words on one line left of `x_limit` are merged before matching."""
    lines: dict[int, list[Word]] = {}
    for w in words:
        if w.x0 < x_limit:
            lines.setdefault(round(w.top), []).append(w)
    found: list[Word] = []
    for ws in lines.values():
        ws.sort(key=lambda w: w.x0)
        # the first date on the line (a Value Date column may follow)
        for n in (3, 1, 2):
            if len(ws) >= n:
                text = " ".join(w.text for w in ws[:n])
                if pattern.fullmatch(text):
                    found.append(Word(text, ws[0].x0, ws[n - 1].x1, ws[0].top, ws[0].page))
                    break
    return found


def reconstruct_rows(doc: PdfDoc, spec: dict[str, tuple[str, ...]],
                     date_pattern: re.Pattern, narration_from: str = "after_date") -> list[RawRow]:
    rows: list[RawRow] = []
    columns: Optional[Columns] = None
    for page_no, words in enumerate(doc.pages):
        found = find_columns(words, spec, narration_from)
        if found is not None:
            columns = found
            start = found.header_bottom
        elif columns is None:
            continue
        else:
            start = 0.0      # table continuing on a page without its header
        cols = columns
        body = [w for w in words if w.top > start]
        anchors = sorted(_date_tokens(body, cols.narration_x0, date_pattern), key=lambda w: w.top)
        if not anchors:
            continue
        # The narration column begins where the date column ends.
        edges = {"debit": cols.debit_x1, "credit": cols.credit_x1, "balance": cols.balance_x1}

        def amount_role(w: Word) -> Optional[str]:
            """An amount is a number right-aligned under an amount header.
            (SBI's description column runs almost up to the Debit column, so
            position alone cannot separate them.)"""
            if parse_amount(w.text) is None:
                return None
            role = min(edges, key=lambda k: abs(edges[k] - w.x1))
            return role if abs(edges[role] - w.x1) <= 30 else None

        narration = [w for w in body if cols.narration_x0 <= w.x0 < max(edges.values())
                     and w.x0 < cols.narration_x1 + 5
                     and not date_pattern.fullmatch(w.text) and amount_role(w) is None]
        gaps = [b.top - a.top for a, b in zip(anchors, anchors[1:])]
        gap = statistics.median(gaps) if gaps else 20.0
        # Centred rows put narration ABOVE their date as often as below it;
        # top-aligned rows never do. Measured over the page, not assumed.
        offsets = [w.top - min(anchors, key=lambda a: abs(a.top - w.top)).top for w in narration]
        above = sum(1 for o in offsets if o < -2)
        centred = bool(offsets) and above / len(offsets) > 0.2

        assigned: dict[int, list[Word]] = {i: [] for i in range(len(anchors))}
        for w in narration:
            if centred:
                i = min(range(len(anchors)), key=lambda k: abs(anchors[k].top - w.top))
                if abs(anchors[i].top - w.top) <= gap * 0.6:
                    assigned[i].append(w)
            else:
                above = [k for k, a in enumerate(anchors) if a.top - 3 <= w.top]
                if above:
                    i = above[-1]
                    # The last row on a page ends one row-height down: further
                    # text is the page footer, not a continuation line.
                    nxt = anchors[i + 1].top - 3 if i + 1 < len(anchors) else anchors[i].top + gap * 1.05
                    if w.top < nxt:
                        assigned[i].append(w)

        for i, anchor in enumerate(anchors):
            line_words = [w for w in body if abs(w.top - anchor.top) <= 3 and w.x0 >= cols.narration_x0]
            debit = credit = balance = None
            ambiguous = False
            for w in line_words:
                value = parse_amount(w.text)
                if value is None:
                    continue
                role = amount_role(w)
                if role is None:
                    if w.x0 >= cols.narration_x1:
                        ambiguous = True
                    continue
                if role == "debit":
                    debit = value
                elif role == "credit":
                    credit = value
                else:
                    balance = value
            lines: dict[int, list[Word]] = {}
            for w in sorted(assigned[i], key=lambda w: (round(w.top), w.x0)):
                lines.setdefault(round(w.top), []).append(w)
            # A line opening with "*" is a statement banner ("**This is a
            # computer generated statement**", "*** End of Statement ***"),
            # never a narration continuation.
            ordered = [ws for _, ws in sorted(lines.items()) if not ws[0].text.startswith("*")]
            text_lines = [" ".join(w.text for w in ws) for ws in ordered]
            rows.append(RawRow(page=page_no, date_text=anchor.text, narration_lines=text_lines,
                               line_right_edges=[max(w.x1 for w in ws) for ws in ordered],
                               debit=debit, credit=credit, balance=balance,
                               amount_ambiguous=ambiguous))
    return rows


def join_wrapped(rows: list[RawRow], rule: str = "width") -> dict[int, str]:
    """Join each row's narration lines, gluing hard wraps.

    Banks wrap narrations two ways: a HARD wrap cuts a token where the column
    ends ("chittibabudar" + "am@axl"), a SOFT wrap breaks at a space
    ("Swiggy" + "Ltd/UTIB/...") or before a "/" ("SYNTHNAME" + "/HDFC/...").
    A line that reaches the column's right edge was cut: glue without a
    space. A break at "/" is glued too. Anything else was a space.
    The right edge is measured (the 90th percentile of first-line ends), not
    assumed, because proportional fonts make character counts meaningless.
    """
    ends = sorted(r.line_right_edges[0] for r in rows if len(r.line_right_edges) > 1)
    limit = ends[int(0.9 * (len(ends) - 1))] if ends else None
    # rule="chars": the bank wraps at a fixed CHARACTER count (Bank of
    # Baroda: 43), so a first line of that length is a hard wrap whatever
    # its rendered width.
    firsts = Counter(len(r.narration_lines[0]) for r in rows if len(r.narration_lines) > 1)
    char_width = firsts.most_common(1)[0][0] if firsts else None
    joined: dict[int, str] = {}
    for idx, r in enumerate(rows):
        text = ""
        for k, line in enumerate(r.narration_lines):
            if k == 0:
                text = line
                continue
            prev, prev_end = r.narration_lines[k - 1], r.line_right_edges[k - 1]
            if rule == "chars":
                hard = char_width is not None and len(prev) >= char_width - 1
            else:
                hard = limit is not None and prev_end >= limit - 4
            # Soft-wrap opportunities that carry no space: after "/" or "-",
            # before "/" or "@" (seen on the real SBI statement:
            # "gpay-" + "12190", "abc" + "@okaxis").
            if hard or line.startswith(("/", "@")) or prev.endswith(("/", "-")):
                text += line
            else:
                text += " " + line
        joined[idx] = text
    return joined


def wrap_cap(narrations: list[str], default: int) -> int:
    """The bank's maximum narration length: the longest length that at least
    two rows share (a cap, not a coincidence). Falls back to the adapter's
    documented default."""
    lengths = Counter(len(n) for n in narrations if n)
    shared = [length for length, count in lengths.items() if count >= 2]
    return max(shared) if shared and max(shared) >= default - 6 else default


def complete_vpa(token: str, followed_by_suffix: bool, at_cap: bool) -> tuple[Optional[str], Optional[str]]:
    """(upi_id, fragment). A VPA is only an identity when it is COMPLETE.

    Complete means: it validates, and either the narration continues after it
    (so it was not cut), or the narration is shorter than the bank's cap, or
    its handle is a known PSP handle that no other known handle extends. A
    truncated token ("veerendragandham9@o", "muralinaga") is returned as a
    fragment - kept for display, never matched against anything.
    """
    from backend.app.services.vpa import KNOWN_HANDLES

    token = (token or "").strip()
    if not token:
        return None, None
    upi = normalize_upi_id(token)
    if upi is None:
        return None, token
    if followed_by_suffix or not at_cap:
        return upi, None
    handle = upi.split("@", 1)[1]
    extended = any(h != handle and h.startswith(handle) for h in KNOWN_HANDLES)
    if handle in KNOWN_HANDLES and not extended:
        return upi, None
    return None, token


def payment_method_of(narration: str) -> Optional[str]:
    head = re.split(r"[/\-: ]", (narration or "").strip(), maxsplit=1)[0].upper()
    for method in ("UPI", "NEFT", "IMPS", "RTGS", "ATM", "POS", "ECS", "NACH", "ACH", "CHQ",
                   "CDM", "CASH"):
        if head.startswith(method):
            return method
    if re.search(r"(?i)\bint\.?\s*pd\b|interest", narration or ""):
        return "INTEREST"
    return None


def direction_from_columns(row: RawRow) -> Optional[Direction]:
    if row.amount_ambiguous:
        return None
    if row.debit and not row.credit:
        return Direction.DEBIT
    if row.credit and not row.debit:
        return Direction.CREDIT
    return None


class ProviderAdapter:
    """One bank's layout and narration grammar."""

    name = "generic"
    columns: dict[str, tuple[str, ...]] = {}
    date_pattern: re.Pattern = re.compile(r"\d{2}-\d{2}-\d{4}")
    date_formats: tuple[str, ...] = ("%d-%m-%Y",)
    default_cap: int = 50
    narration_from: str = "after_date"     # or "header": narration starts at its header
    wrap_rule: str = "width"               # or "chars" for fixed character-count wrapping

    def detect(self, doc: PdfDoc) -> Detection:  # pragma: no cover - abstract
        raise NotImplementedError

    def parse_narration(self, narration: str, at_cap: bool) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def parse_date(self, text: str) -> Optional[datetime]:
        for fmt in self.date_formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def parse(self, doc: PdfDoc) -> list[CanonicalTransaction]:
        rows = reconstruct_rows(doc, self.columns, self.date_pattern, self.narration_from)
        joined = join_wrapped(rows, self.wrap_rule)
        cap = wrap_cap(list(joined.values()), self.default_cap)
        out: list[CanonicalTransaction] = []
        for idx, row in enumerate(rows):
            narration = joined[idx]
            if row.debit is None and row.credit is None:
                # Opening/closing balance lines carry only a balance: not a
                # transaction, and not a rejected one either.
                continue
            info = self.parse_narration(narration, at_cap=len(narration) >= cap)
            date = self.parse_date(row.date_text)
            tx = CanonicalTransaction(source_provider=self.name, source_format="pdf",
                                      source_row=idx)
            clock = info.get("time")
            if date is not None and clock:
                tx.timestamp = datetime.combine(date.date(), clock)
                tx.time_known = True
                tx.timestamp_confidence = "HIGH"
            elif date is not None:
                tx.timestamp = date
                tx.time_known = False
                tx.timestamp_confidence = "MEDIUM"
            else:
                tx.timestamp_confidence = "LOW"
            tx.direction = direction_from_columns(row)
            tx.amount = row.debit if tx.direction is Direction.DEBIT else (
                row.credit if tx.direction is Direction.CREDIT else (row.debit or row.credit))
            narrated = info.get("narrated_direction")
            if narrated and tx.direction and narrated is not tx.direction:
                tx.reasons.append("direction_conflict")
                tx.direction = None
            tx.balance = row.balance
            tx.narration = narration
            tx.transaction_id = info.get("reference_id")
            tx.counterparty_name = info.get("counterparty_name")
            tx.counterparty_upi_id = info.get("counterparty_upi_id")
            tx.upi_fragment = info.get("upi_fragment")
            tx.counterparty_identifier = info.get("counterparty_identifier")
            tx.identifier_type = info.get("identifier_type")
            tx.name_source = info.get("name_source")
            tx.name_confidence = info.get("name_confidence")
            tx.upi_confidence = info.get("upi_confidence")
            tx.raw_narration = "|".join(row.narration_lines) or None
            tx.counterparty_bank_code = info.get("counterparty_bank_code")
            tx.payment_method = info.get("payment_method") or payment_method_of(narration)
            tx.transaction_type = info.get("transaction_type") or tx.payment_method
            tx.extraction_source = f"{self.name}:{info.get('grammar', 'unparsed')}"
            tx.extraction_confidence = info.get("confidence", 0.5)
            tx.confidence = tx.extraction_confidence
            if tx.counterparty_upi_id is None and tx.counterparty_name is None \
                    and info.get("display_label"):
                tx.counterparty_display_label = info["display_label"]
                tx.display_label_source = info.get("display_label_source")
            if tx.counterparty_identifier and not tx.counterparty_upi_id:
                # Informational, not a defect: the statement printed a partial
                # or non-VPA identifier, which is kept but not matched.
                tx.reasons.append("identifier_not_complete_vpa")
            if row.amount_ambiguous:
                tx.reasons.append("amount_column_ambiguous")
            out.append(tx)
        return out


def structural_score(doc: PdfDoc, checks: list[tuple[float, str, Callable[[PdfDoc], bool]]],
                     provider: str) -> Detection:
    score, evidence = 0.0, []
    for weight, label, test in checks:
        if test(doc):
            score += weight
            evidence.append(label)
    return Detection(provider, round(min(score, 1.0), 2), evidence)
