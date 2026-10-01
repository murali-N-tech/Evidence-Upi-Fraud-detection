import type { RiskLevel, Verdict } from "./lib/verdict";

// `Verdict` is the canonical risk band and `RiskLevel` the legacy
// LOW / MEDIUM / HIGH label the older panels still render. Both come from
// lib/verdict so no view can spell a band its own way.

export interface Transaction {
  transaction_id: string;
  amount: number;
  device_score: number;
  location_score: number;
  velocity_score: number;
  sender: string;
  receiver: string;
  timestamp: string;
  // The canonical band for this row, decided by the server.
  verdict?: Verdict;
  // Legacy flag the monitor table renders. The server derives it from
  // `verdict`, so the two cannot disagree.
  risk: number;
  risk_score: number;
}

export interface BehaviorProfile {
  user_id: string;
  transaction_count: number;
  debit_count?: number;
  credit_count?: number;
  avg_amount: number;
  median_amount?: number;
  max_amount: number;
  min_amount: number;
  most_active_hour: number | null;
  night_transactions: number;
  /** Rows that carried a real clock time. The denominator for night_transactions. */
  timed_transaction_count?: number;
  /** Every distinct payee, not the top 10 merchant_frequency holds. */
  distinct_payees?: number;
  weekend_transactions: number;
  transactions_without_time?: number;
  favorite_merchants: string[];
  average_daily_transactions: number;
  failed_transactions: number;
  known_upi_ids: string[];
  merchant_frequency: Record<string, number>;
  hourly_distribution: Record<string, number>;
  monthly_totals: Record<string, number>;
  source_type: string;
  updated_at: string;
}

export interface PersonalizedAssessment {
  risk_score: number;
  // This family's own sub-score in the canonical vocabulary. It is not the
  // payment's verdict - that comes from the payee check, once every family has
  // been combined.
  band?: Verdict;
  // Legacy label, derived from `band` server-side.
  risk_level: RiskLevel;
  reasons: string[];
  comparison: {
    average_amount?: number;
    max_amount?: number;
    most_active_hour?: number | null;
    average_daily_transactions?: number;
    amount_multiple?: number;
    projected_daily_transactions?: number;
  };
  profile_available: boolean;
  timestamp: string;
  merchant: string;
  location: string | null;
}

export interface StatementTransactionsPage {
  total: number;
  limit: number;
  offset: number;
  truncated: boolean;
  transactions: StatementTransaction[];
}

export interface StatementTransaction {
  statement_id: string;
  user_id: string;
  timestamp: string;
  amount: number;
  merchant: string;
  upi_id?: string;
  status: string;
  reference_number?: string;
  source_type: string;
  txn_type?: string;
  direction?: "DEBIT" | "CREDIT" | null;
  counterparty_name?: string | null;
  counterparty_upi_id?: string | null;
  validation_status?: "VALID" | "PARTIAL" | null;
  validation_reasons?: string | null;
  created_at: string;
}

/** Counts only - the upload no longer returns transaction rows. */
export interface StatementImportReport {
  status: "COMPLETE" | "PARTIAL" | "FAILED";
  source_format: string;
  source_provider?: string | null;
  total_rows_detected: number;
  valid_transactions: number;
  partial_transactions: number;
  rejected_transactions: number;
  duplicate_transactions: number;
  cross_source_duplicates?: number;
  ambiguous_cross_source_kept?: number;
  repeated_rows_in_file: number;
  stored_transactions: number;
  debit_transactions: number;
  credit_transactions: number;
  unknown_direction: number;
  missing_timestamp: number;
  future_timestamp: number;
  malformed_upi_id: number;
  upi_linked_transactions: number;
  upi_transactions?: number;
  counterparty_vpas_found?: number;
  distinct_counterparty_vpas?: number;
  counterparty_names_found?: number;
  missing_identity?: number;
  truncated_upi_ids?: number;
  reversal_transactions?: number;
  complete_vpas_found?: number;
  partial_identifiers_found?: number;
  non_vpa_identifiers_found?: number;
  reference_ids_found?: number;
  no_identity_rows?: number;
  identifier_types?: Record<string, number>;
  wrapped_narrations_reconstructed?: number;
  identity_states?: Record<string, number>;
  provider_confidence?: number | null;
  provider_evidence?: string[];
  debits_without_upi_id: number;
  payee_relationships_updated: number;
  graph_observations_updated: number;
  undated_payments_recorded: number;
  profile_updated: boolean;
  date_range: { from: string | null; to: string | null };
  unmapped_columns: string[];
  warnings: string[];
}

export type PayeeStatus =
  | "PAID_BEFORE" | "PAID_ONCE" | "PAID_OFTEN" | "NEW_TO_USER_BUT_OBSERVED"
  | "UNKNOWN" | "IDENTITY_UNAVAILABLE" | "HISTORY_UNAVAILABLE_DUE_TO_PARTIAL_IMPORT"
  | "PARTIAL_IDENTITY" | "NAME_ONLY_HISTORY";

export interface PayeeHistorySummary {
  upi_id: string;
  paid_before: boolean;
  payment_count: number;
  dated_payment_count: number;
  first_payment_at: string | null;
  last_payment_at: string | null;
  total_amount: number;
  days_since_first_payment: number | null;
  days_since_last_payment: number | null;
  relationship_status: string | null;
  sources: Record<string, number>;
}

export interface PayeeFinding {
  code: string;
  severity: "info" | "warn" | "high" | "critical";
  message: string;
}

export interface PayeeReputation {
  vpa: string;
  known: boolean;
  display_name: string | null;
  first_seen: string | null;
  last_seen: string | null;
  age_days: number | null;
  /** Amount/time fields are null when fewer than 5 other payers exist. */
  payment_count: number | null;
  distinct_payers: number;
  repeat_payers: number;
  total_amount: number | null;
  detail_withheld?: boolean;
  mean_amount: number | null;
  amount_spread: number | null;
  reports: number;
  blocked: boolean;
  /** Distinct payers per source. Observations, never "verified". */
  observed_by?: Record<"SYSTEM_CONFIRMED" | "STATEMENT_OBSERVED" | "OTHER_OBSERVED", number>;
}

/** One evidence family's row, as the payee check publishes it. The payer's own
 *  assessment sits under `detail`; the family's contribution to the combined
 *  score is `score`, on a different scale, and the two are not interchangeable. */
export interface PayerBehaviourEvidence {
  available: boolean;
  score: number | null;
  severity: string | null;
  code: string;
  message?: string | null;
  facts?: string[];
  detail: PersonalizedAssessment | null;
}

export interface PayeeCheckResult {
  input: string;
  input_kind: string;
  payee: {
    vpa: string | null;
    key: string;
    display_name: string | null;
    valid: boolean;
    handle_type: string;
    impersonates: string | null;
  };
  request: {
    amount: number | null;
    amount_locked: boolean;
    note: string | null;
    merchant_code: string | null;
    signed: boolean;
  };
  reputation: PayeeReputation | null;
  /** The signed-in payer's own relationship with this payee. */
  payee_status?: PayeeStatus;
  history_state?: { state: string; message: string };
  payee_history?: PayeeHistorySummary | null;
  risk_score: number;
  // The canonical classification. `decision` is the same value under the name
  // the first version of this screen read, and is kept only for that.
  verdict: Verdict;
  decision: Verdict;
  headline: string;
  component_scores: {
    address_and_qr: number;
    payee_history: number;
    amount_context: number;
    stated_intent: number;
    message_pressure: number;
    link_safety: number;
  };
  intent: {
    supplied: boolean;
    intent: string | null;
    label: string | null;
    score: number;
    findings: PayeeFinding[];
  };
  message_pressure: {
    supplied: boolean;
    score: number;
    probability: number;
    patterns: Record<string, number>;
    language_note: string | null;
    model: string;
    weak_only: boolean;
    findings: { code: string; severity: string; message: string; quote: string | null }[];
  };
  links: {
    found: number;
    score: number;
    links: {
      url: string;
      host: string;
      registrable: string;
      claimed_brand: string | null;
      score: number;
      features: Record<string, number>;
      findings: PayeeFinding[];
    }[];
  };
  // What corroborated, and who saw each thing. `facts` are the distinct
  // observations; `observed_by` maps each one to the families that reported it.
  // This used to be `{ families: string[] }` - the fact-aware corroboration
  // change renamed it server-side and this type was not updated, so the screen
  // read `agreement.families.length` off undefined and crashed on exactly the
  // payments that corroborate.
  agreement: {
    facts: string[];
    bonus: number;
    observed_by?: Record<string, string[]>;
  };
  findings: PayeeFinding[];
  payer_behaviour: PayerBehaviourEvidence | null;
}

export interface IntentOption {
  id: string;
  label: string;
}
