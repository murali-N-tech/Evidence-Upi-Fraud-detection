"""Clear raw statement text stored by earlier versions (opt-in).

New imports no longer write `statement_transactions.raw_line`. Rows imported
before that change still carry the raw statement line. This script counts
them and, only with --apply, sets the column to NULL. Nothing else changes:
the parsed fields, profiles and payee history are untouched.

    python scripts/purge_statement_raw_text.py            # count only
    python scripts/purge_statement_raw_text.py --apply    # clear raw_line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.app.services.profile_store as store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually clear raw_line")
    parser.add_argument("--db", help="database path (default: data/behavior_profiles.db)")
    args = parser.parse_args()
    if args.db:
        store.DB_PATH = Path(args.db)
    conn = store._get_connection()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM statement_transactions WHERE raw_line IS NOT NULL"
        ).fetchone()[0]
        print(f"rows with stored raw statement text: {n}")
        if args.apply and n:
            with conn:
                conn.execute("UPDATE statement_transactions SET raw_line = NULL")
            print("cleared.")
        elif n:
            print("re-run with --apply to clear them.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
