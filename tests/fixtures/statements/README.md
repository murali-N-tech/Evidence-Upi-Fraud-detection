# Statement fixtures - SYNTHETIC

Every file here is synthetic. The layouts are *like* the named providers'
exports (column names and conventions chosen to resemble them); none is a real
export, and passing these tests does not mean every real GPay / PhonePe /
Paytm / bank statement is supported.

* `gpay_like.csv` - Date + Time columns, Paid/Received type, UPI ID column
* `phonepe_like.csv` - DEBIT/CREDIT type column, VPA only inside the details text
* `paytm_like.csv` - single signed amount column (negative = paid)
* `bank_like.csv` - account preamble above the header, Withdrawal/Deposit
  columns, dd/mm/yy dates, VPA inside the narration
* `demo_appended_rows.csv` - rows appended to `docs/demo/DEMO_UPI_Statement.csv`
  to reproduce the reported bug. The 9063453458@axl row is the exact row from
  the bug report (kept for the regression test, fixture use only). The other
  appended rows were replaced with synthetic payees; the DEBET misspelling
  is kept on purpose.

PDF fixtures are generated inside the tests (reportlab), so no binary is
committed: a PhonePe-like app layout (scripts/make_demo_statement.render_pdf),
a GPay-like text layout and a bank-like multi-page table with repeated
headers and wrapped narrations.
