"""Synthetic PDF statements, generated at test time (no binaries committed).

Each layout is LIKE a provider's export, not a copy of one. See
tests/fixtures/statements/README.md.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def have_rupee_font() -> bool:
    return DEJAVU.exists()


def bank_like_table_pdf(rows: list[tuple[str, str, str, str, str, str]],
                        rows_per_page: int = 25) -> bytes:
    """Multi-page ruled table; the header row repeats on every page and long
    narrations wrap inside their cell."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

    style = getSampleStyleSheet()["BodyText"]
    style.fontSize = 7
    style.leading = 8
    header = ["Txn Date", "Narration", "Ref No", "Withdrawal", "Deposit", "Balance"]
    story = [Paragraph("SYNTHETIC BANK-LIKE STATEMENT - not a real export", style)]
    for start in range(0, len(rows), rows_per_page):
        chunk = rows[start:start + rows_per_page]
        data = [header] + [[r[0], Paragraph(r[1], style), r[2], r[3], r[4], r[5]] for r in chunk]
        table = Table(data, colWidths=[55, 200, 80, 55, 55, 60], repeatRows=1)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                                   ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story += [table, Paragraph("Page footer - SYNTHETIC BANK - customer care 1800-000-000", style),
                  PageBreak()]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()


def gpay_like_pdf(rows: list[tuple[datetime, str, str, float, str]]) -> bytes:
    """(when, 'Paid to'|'Received from', name, amount, upi_txn_id) blocks in a
    Google-Pay-like text layout. Needs a font with the rupee sign."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("DejaVu", str(DEJAVU)))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("DejaVu", 9)
    y = 800
    c.drawString(40, y, "Transaction statement")
    y -= 14
    c.drawString(40, y, "SYNTHETIC GPay-like layout - not a real export")
    y -= 24
    for when, verb, name, amount, txn_id in rows:
        if y < 80:
            c.showPage()
            c.setFont("DejaVu", 9)
            y = 800
        c.drawString(40, y, f"{when:%d%b,%Y} {verb.replace(' ', '')}{name} ₹{amount:,.2f}")
        c.drawString(40, y - 11, f"{when:%I:%M%p}")
        c.drawString(40, y - 22, f"UPITransactionID:{txn_id}")
        y -= 40
    c.save()
    return buf.getvalue()


def phonepe_like_pdf(rows: list[dict], tmp_dir: Path) -> bytes:
    """The app-block layout from scripts/make_demo_statement.py."""
    import random
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import make_demo_statement as demo  # noqa: E402

    txs = [demo.Tx(when=r["when"], counterparty=r["name"], vpa=r["vpa"],
                   amount=r["amount"], direction=r["direction"]) for r in rows]
    out = tmp_dir / "phonepe_like.pdf"
    demo.render_pdf(txs, out, random.Random(7))
    return out.read_bytes()


# ── Bank statement layouts (sanitised; modelled on real statements' structure) ─

A4_HEIGHT = 842.0


def bob_like_pdf(rows: list[dict], wrap: int = 43, cap: int = 49,
                 rows_per_page: int = 30) -> bytes:
    """Bank-of-Baroda-like account statement.

    Structure copied from a real statement: DATE | NARRATION | CHQ.NO. |
    WITHDRAWAL (DR) | DEPOSIT (CR) | BALANCE; rows vertically centred, the
    narration hard-wrapped at `wrap` characters and capped at `cap`. Values
    are synthetic. rows: {date:'dd-mm-yyyy', narration, dr|cr: float, bal}.
    """
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595.0, A4_HEIGHT))

    def header() -> float:
        c.setFont("Helvetica", 7)
        c.drawString(40, A4_HEIGHT - 60, "SYNTHETIC TEST ACCOUNT - Bank of Baroda-like layout")
        c.drawString(40, A4_HEIGHT - 72, "IFSC BARB0SYNTH1  www.bankofbaroda.in")
        y = A4_HEIGHT - 110
        for x, text in ((43, "DATE"), (141, "NARRATION"), (252, "CHQ.NO."),
                        (308, "WITHDRAWAL (DR)"), (423, "DEPOSIT (CR)"), (530, "BALANCE")):
            c.drawString(x, y, text)
        return y

    y = header()
    count = 0
    opening = True
    for r in [{"date": rows[0]["date"], "narration": "Opening Balance", "bal": 1000.0}] + rows:
        if count and count % rows_per_page == 0:
            c.drawString(40, 40, "Page footer - synthetic")
            c.showPage()
            y = header()
        y -= 21
        text = r["narration"][:cap]
        lines = [text] if (opening or len(text) <= wrap) else [text[:wrap], text[wrap:]]
        opening = False
        c.drawString(33, y, r["date"])
        if len(lines) == 1:
            c.drawString(80, y, lines[0])
        else:
            c.drawString(80, y + 5, lines[0])
            c.drawString(80, y - 5, lines[1])
        if r.get("dr") is not None:
            c.drawRightString(378, y, f"{r['dr']:.2f}")
        if r.get("cr") is not None:
            c.drawRightString(471, y, f"{r['cr']:.2f}")
        c.drawRightString(555, y, f"{r.get('bal', 500.0):.2f}")
        c.drawString(557, y, "Cr")
        count += 1
    c.drawString(40, 60, "Abbreviations: UPI - Unified Payments Interface  (legend text)")
    c.save()
    return buf.getvalue()


def sbi_like_pdf(rows: list[dict], wrap_width: float = 190.0) -> bytes:
    """State-Bank-of-India-like account statement.

    Structure copied from a real SBI statement: Txn Date | Value Date |
    Description | Debit | Credit | Balance, dd/mm/yyyy dates, TOP-aligned
    rows, the description wrapped by rendered width (at a space, before a
    "/", or mid-token at the column edge), and a "**...**" line after the
    last row. Values are synthetic. rows: {date:'dd/mm/yyyy', narration,
    dr|cr, bal}.
    """
    from reportlab.pdfgen import canvas

    font, size = "Helvetica", 7
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595.0, A4_HEIGHT))
    c.setFont(font, size)
    width = lambda t: c.stringWidth(t, font, size)

    def wrap(text: str) -> list[str]:
        lines, line = [], ""
        i = 0
        while i < len(text):
            ch = text[i]
            if width(line + ch) > wrap_width:
                # break at the last space or before the last "/", else hard
                cut = max(line.rfind(" "), line.rfind("/"))
                if cut > 0 and line[cut] == " ":
                    lines.append(line[:cut]); line = line[cut + 1:]
                elif cut > 0:
                    lines.append(line[:cut]); line = line[cut:]
                else:
                    lines.append(line); line = ""
                continue
            line += ch
            i += 1
        return lines + ([line] if line else [])

    c.drawString(40, A4_HEIGHT - 50, "SYNTHETIC TEST ACCOUNT - State Bank of India-like layout")
    c.drawString(40, A4_HEIGHT - 62, "IFSC SBIN0SYNTH1")
    y = A4_HEIGHT - 100
    heads = ((49, "Txn Date"), (110, "Value Date"), (240, "Description"),
             (384, "Debit"), (447, "Credit"), (509, "Balance"))
    for x, text in heads:
        c.drawString(x, y, text)
    right = {"Debit": 387, "Credit": 463, "Balance": 533}
    y -= 14
    for r in rows:
        lines = wrap(r["narration"]) or [""]
        c.drawString(38, y, r["date"])
        c.drawString(103, y, r["date"])
        for k, line in enumerate(lines):
            c.drawString(169, y - 10 * k, line)
        if r.get("dr") is not None:
            c.drawRightString(right["Debit"], y, f"{r['dr']:.2f}")
        if r.get("cr") is not None:
            c.drawRightString(right["Credit"], y, f"{r['cr']:.2f}")
        c.drawRightString(right["Balance"], y, f"{r.get('bal', 500.0):.2f}")
        y -= 10 * max(len(lines), 2) + 4
    c.drawString(169, y - 2, "**This is a computer generated statement (synthetic).**")
    c.save()
    return buf.getvalue()
