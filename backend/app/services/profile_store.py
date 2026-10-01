from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "behavior_profiles.db"


# Schema setup ran on every single connection, which meant a CREATE TABLE, four
# CREATE INDEX statements and a commit on every request - write traffic for a
# read, and lock churn as soon as two processes talk to the file. It only needs
# to happen once per database.
#
# Keyed by path rather than a single flag: the maintenance scripts point
# DB_PATH at another file with --db, and tests point it at a temp directory. A
# boolean would let the second database go without a schema.
_SCHEMA_DONE: set[str] = set()


def _get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Two processes share this file (Express forwards to FastAPI, and the
    # scripts run alongside), so wait for a lock rather than failing instantly.
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")

    key = str(Path(DB_PATH).resolve())
    if key in _SCHEMA_DONE:
        return conn

    _ensure_schema(conn)
    _SCHEMA_DONE.add(key)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS statement_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp TEXT,
            amount REAL NOT NULL,
            merchant TEXT,
            upi_id TEXT,
            status TEXT,
            reference_number TEXT,
            source_type TEXT,
            raw_line TEXT,
            txn_type TEXT NOT NULL DEFAULT 'DEBIT',
            time_known INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )
    # Older databases predate txn_type; add it rather than losing their rows.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(statement_transactions)")}
    if "txn_type" not in columns:
        conn.execute(
            "ALTER TABLE statement_transactions ADD COLUMN txn_type TEXT NOT NULL DEFAULT 'DEBIT'"
        )
    if "time_known" not in columns:
        # Whether the statement actually carried a clock time. Without it a
        # parser placeholder is indistinguishable from a real payment at that
        # hour, and the profile reports a confident "most active hour" that
        # was never in the data.
        conn.execute(
            "ALTER TABLE statement_transactions ADD COLUMN time_known INTEGER NOT NULL DEFAULT 1"
        )
    # Canonical-ingestion columns. Additive: rows stored before them keep
    # working and read these as NULL.
    for name, decl in (
        ("direction", "TEXT"),                 # DEBIT / CREDIT / NULL (unknown)
        ("counterparty_name", "TEXT"),
        ("counterparty_upi_id", "TEXT"),       # normalize_upi_id() output only
        ("transaction_id", "TEXT"),
        ("source_provider", "TEXT"),
        ("source_format", "TEXT"),
        ("confidence", "REAL"),
        ("validation_status", "TEXT"),         # VALID / PARTIAL
        ("validation_reasons", "TEXT"),
        ("fingerprint", "TEXT"),
        # Bank-statement adapters (backend/app/ingestion/providers).
        ("identity_state", "TEXT"),            # EXACT_VPA / VPA_ONLY / NAME_ONLY / UNKNOWN
        ("timestamp_confidence", "TEXT"),
        ("payment_method", "TEXT"),
        ("transaction_type", "TEXT"),
        ("counterparty_bank_code", "TEXT"),
        ("balance", "REAL"),
        ("extraction_source", "TEXT"),
        ("counterparty_identifier", "TEXT"),   # kept even when not a complete VPA
        ("identifier_type", "TEXT"),           # COMPLETE_VPA / PARTIAL_VPA_FRAGMENT / ...
        ("name_source", "TEXT"),
        ("name_confidence", "REAL"),
        ("upi_confidence", "REAL"),
        ("narration", "TEXT"),                 # reconstructed narration (user's own)
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE statement_transactions ADD COLUMN {name} {decl}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS statement_imports (
            statement_id TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            status       TEXT NOT NULL,      -- COMPLETE / PARTIAL / FAILED
            report_json  TEXT NOT NULL       -- counts only, never row contents
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_imports_user ON statement_imports(user_id, created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS behavior_profiles (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            transaction_count INTEGER NOT NULL,
            source_type TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            upi_id TEXT,
            upi_verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    # Older databases have a users table without the UPI columns. This has to
    # run after the CREATE above, or a brand new database fails on a table
    # that does not exist yet.
    user_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "upi_id" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN upi_id TEXT")
    if "upi_verified" not in user_columns:
        # 0 means nobody confirmed the address exists. It is never set to 1
        # unless a verification provider actually said so.
        conn.execute("ALTER TABLE users ADD COLUMN upi_verified INTEGER NOT NULL DEFAULT 0")
    # H3: every profile read and every statement listing filtered on user_id
    # with no index, so each one scanned the whole table.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_stmt_user ON statement_transactions(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_stmt_user_ts ON statement_transactions(user_id, timestamp)"
    )

    # C4: the natural key for a statement line. With this in place,
    # INSERT OR IGNORE makes re-uploading the same statement a no-op.
    # Skipped when the table still holds duplicates from before the fix -
    # scripts/rebuild_profiles.py cleans those up, and the index is created
    # on the next connection.
    #
    # COALESCE, not the bare columns. SQLite treats every NULL as distinct in a
    # UNIQUE index, so the original index suppressed nothing for the rows that
    # need it most: a statement line with no reference number (common in PDF
    # statements) compared unequal to an identical stored row, and every
    # re-upload inserted the whole file again. Measured: three inserts of the
    # same NULL-reference row produced three rows under the old index and one
    # under this one.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_stmt_user_upi "
        "ON statement_transactions(user_id, counterparty_upi_id)"
    )
    # The dedupe rule is now the fingerprint (backend/app/ingestion/dedupe.py).
    # ux_stmt_tx_v2 keyed on (time, amount, merchant, reference) and so merged
    # two different payees paid the same amount in the same minute, and
    # silently dropped the second of two genuinely identical payments.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_stmt_fingerprint "
        "ON statement_transactions(user_id, fingerprint) WHERE fingerprint IS NOT NULL"
    )
    conn.execute("DROP INDEX IF EXISTS ux_stmt_tx_v2")
    conn.execute("DROP INDEX IF EXISTS ux_stmt_tx")

    conn.commit()
    return conn


_STMT_COLUMNS = (
    "statement_id", "user_id", "timestamp", "amount", "merchant", "upi_id",
    "status", "reference_number", "source_type", "raw_line", "txn_type",
    "time_known", "created_at", "direction", "counterparty_name",
    "counterparty_upi_id", "transaction_id", "source_provider", "source_format",
    "confidence", "validation_status", "validation_reasons", "fingerprint",
    "identity_state", "timestamp_confidence", "payment_method", "transaction_type",
    "counterparty_bank_code", "balance", "extraction_source",
    "counterparty_identifier", "identifier_type", "name_source", "name_confidence",
    "upi_confidence", "narration",
)


def insert_statement_rows(
    user_id: str,
    statement_id: str,
    source_type: str,
    transactions: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> list[int]:
    """Insert rows; return the positions (in `transactions`) actually stored.

    A row whose fingerprint is already stored for this user is the same
    transaction imported again and is skipped. The returned list is what the
    caller may count as NEW observations - which is how re-uploading a
    statement is kept from inflating payee history.

    `raw_line` is no longer written. The canonical fields carry everything
    the system uses; keeping the statement's raw text only kept private
    content around for no purpose.
    """
    from backend.app.ingestion.dedupe import fingerprint_dicts

    # Copies: the caller's dicts are not ours to annotate.
    transactions = [dict(tx) for tx in transactions]
    fingerprint_dicts(user_id, transactions)
    created_at = datetime.utcnow().isoformat()
    own = conn is None
    conn = conn or _get_connection()
    placeholders = ", ".join("?" for _ in _STMT_COLUMNS)
    sql = (f"INSERT OR IGNORE INTO statement_transactions ({', '.join(_STMT_COLUMNS)}) "
           f"VALUES ({placeholders})")
    inserted: list[int] = []
    try:
        with conn:
            for position, tx in enumerate(transactions):
                txn_type = str(tx.get("txn_type") or tx.get("direction") or "UNKNOWN").upper()
                cur = conn.execute(sql, (
                    statement_id, user_id, tx.get("timestamp"), float(tx.get("amount") or 0),
                    tx.get("merchant"), tx.get("upi_id"), tx.get("status"),
                    tx.get("reference_number"), source_type, None, txn_type,
                    1 if tx.get("time_known", True) else 0, created_at,
                    tx.get("direction", txn_type if txn_type in ("DEBIT", "CREDIT") else None),
                    tx.get("counterparty_name"), tx.get("counterparty_upi_id"),
                    tx.get("transaction_id"), tx.get("source_provider"),
                    tx.get("source_format"), tx.get("confidence"),
                    tx.get("validation_status"), tx.get("validation_reasons"),
                    tx["fingerprint"],
                    tx.get("identity_state"), tx.get("timestamp_confidence"),
                    tx.get("payment_method"), tx.get("transaction_type"),
                    tx.get("counterparty_bank_code"), tx.get("balance"),
                    tx.get("extraction_source"),
                    tx.get("counterparty_identifier"), tx.get("identifier_type"),
                    tx.get("name_source"), tx.get("name_confidence"),
                    tx.get("upi_confidence"), tx.get("narration"),
                ))
                if cur.rowcount:
                    inserted.append(position)
    finally:
        if own:
            conn.close()
    return inserted


def save_statement_transactions(
    user_id: str,
    statement_id: str,
    source_type: str,
    transactions: list[dict[str, Any]],
) -> int:
    """Kept for existing callers. Returns how many rows were newly stored."""
    return len(insert_statement_rows(user_id, statement_id, source_type, transactions))


def save_import_report(user_id: str, statement_id: str, status: str,
                       report: dict[str, Any]) -> None:
    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO statement_imports "
                "(statement_id, user_id, created_at, status, report_json) VALUES (?,?,?,?,?)",
                (statement_id, user_id, datetime.utcnow().isoformat(), status,
                 json.dumps(report)),
            )
    finally:
        conn.close()


def latest_import(user_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """The user's most recent import report (counts only), or None."""
    own = conn is None
    conn = conn or _get_connection()
    try:
        try:
            row = conn.execute(
                "SELECT statement_id, created_at, status, report_json FROM statement_imports "
                "WHERE user_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    finally:
        if own:
            conn.close()
    if row is None:
        return None
    return {"statement_id": row["statement_id"], "created_at": row["created_at"],
            "status": row["status"], "report": json.loads(row["report_json"])}


def get_user_transactions(user_id: str) -> pd.DataFrame:
    """This user's stored statement rows - their own data only.

    A row flagged `timestamp_in_future` comes back with a NULL timestamp, so
    no temporal consumer (velocity, profile hours, payee dates) can read a
    date that cannot have happened before the import. The raw statement line
    is not returned.
    """
    conn = _get_connection()
    query = """
        SELECT
            statement_id,
            user_id,
            CASE WHEN COALESCE(validation_reasons, '') LIKE '%timestamp_in_future%'
                 THEN NULL ELSE timestamp END AS timestamp,
            amount,
            merchant,
            upi_id,
            status,
            reference_number,
            source_type,
            txn_type,
            time_known,
            created_at,
            direction,
            counterparty_name,
            counterparty_upi_id,
            validation_status,
            validation_reasons
        FROM statement_transactions
        WHERE user_id = ?
        ORDER BY COALESCE(timestamp, created_at)
    """
    df = pd.read_sql_query(query, conn, params=[user_id])
    conn.close()
    return df


def save_behavior_profile(
    user_id: str,
    source_type: str,
    profile: dict[str, Any],
    transaction_count: int,
) -> None:
    conn = _get_connection()
    updated_at = datetime.utcnow().isoformat()

    with conn:
        conn.execute(
            """
            INSERT INTO behavior_profiles (
                user_id,
                profile_json,
                transaction_count,
                source_type,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                transaction_count = excluded.transaction_count,
                source_type = excluded.source_type,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(profile),
                transaction_count,
                source_type,
                updated_at,
            ),
        )

    conn.close()


def get_behavior_profile(user_id: str) -> dict[str, Any] | None:
    conn = _get_connection()
    row = conn.execute(
        """
        SELECT user_id, profile_json, transaction_count, source_type, updated_at
        FROM behavior_profiles
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    profile = json.loads(row["profile_json"])
    profile["user_id"] = row["user_id"]
    profile["transaction_count"] = row["transaction_count"]
    profile["source_type"] = row["source_type"]
    profile["updated_at"] = row["updated_at"]
    return profile

def get_all_edges() -> list[dict[str, str]]:
    conn = _get_connection()
    edges = []
    try:
        with conn:
            cursor = conn.execute(
                "SELECT DISTINCT user_id, merchant FROM statement_transactions WHERE merchant IS NOT NULL AND merchant != 'UNKNOWN_MERCHANT' LIMIT 1000"
            )
            rows = cursor.fetchall()
            for row in rows:
                edges.append({
                    "user": row["user_id"],
                    "merchant": row["merchant"]
                })
        return edges
    except Exception:
        return []

def create_user(
    user_id: str,
    username: str,
    password_hash: str,
    upi_id: str | None = None,
    upi_verified: bool = False,
) -> bool:
    conn = _get_connection()
    created_at = datetime.utcnow().isoformat()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, upi_id, upi_verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, username, password_hash, upi_id, 1 if upi_verified else 0, created_at),
            )
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username: str) -> dict[str, Any] | None:
    conn = _get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, upi_id, upi_verified, created_at "
        "FROM users WHERE username = ? COLLATE NOCASE",
        (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return None
    return dict(row)
