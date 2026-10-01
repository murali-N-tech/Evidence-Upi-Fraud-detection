import React, { useState } from "react";
import { Upload, AlertCircle, FileText, CheckCircle2, ArrowRight, Loader2, FileUp, Info } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import type { StatementImportReport } from "../types";

interface StatementProfilingProps {
  onNavigate?: (page: string) => void;
}

export default function StatementProfiling({ onNavigate }: StatementProfilingProps) {
  const { api, token } = useAuth();
  const [retainSource, setRetainSource] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  const [extractedCount, setExtractedCount] = useState<number>(0);
  const [sourceType, setSourceType] = useState<string>("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [report, setReport] = useState<StatementImportReport | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      const ext = dropped.name.toLowerCase().split(".").pop();
      if (ext === "pdf" || ext === "csv") {
        setFile(dropped);
        setError(null);
      } else {
        setError("Only PDF and CSV files are supported.");
      }
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) {
      setError("Your session has expired. Please sign in again.");
      return;
    }
    if (!file) {
      setError("Please upload a CSV or PDF statement file");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);
    setWarnings([]);
    setReport(null);

    // No user_id: the server takes the owner from the auth token.
    const formData = new FormData();
    formData.append("retain_source", String(retainSource));
    formData.append("file", file);

    try {
      const result = await api<{
        profile?: unknown;
        transactions_extracted?: number;
        source_type?: string;
        warnings?: string[];
        import?: StatementImportReport;
      }>("/api/statement/upload", { method: "POST", body: formData });
      setReport(result.import ?? null);
      if (result.profile || (result.import?.valid_transactions ?? 0) + (result.import?.partial_transactions ?? 0) > 0) {
        setSuccess(true);
        setExtractedCount(result.transactions_extracted || 0);
        setSourceType(result.source_type || "");
        setWarnings(result.warnings || []);
      } else {
        setError("Could not extract any transactions from the uploaded file.");
        setWarnings(result.warnings || []);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in" id="statement-profiling-container">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-ink mb-2">Upload Statement</h1>
        <p className="text-ink-muted">
          Upload a UPI or bank statement (PDF/CSV) to extract transactional features and synthesize your personalized payment behavior profile.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Form Card */}
        <div className="bg-surface border border-line rounded-xl p-6 h-fit" id="upload-card">
          <h2 className="text-xl font-semibold text-ink mb-1">Generate Behaviour Profile</h2>
          <p className="text-xs text-ink-subtle mb-5">CSV with recognisable column headers (date, amount or debit/credit, direction, UPI ID), and text-based PDF statements. Validated on synthetic fixtures and a small set of real PhonePe/GPay PDF exports; real app PDFs name payees without UPI IDs, so they cannot be matched to a scanned QR. Scanned PDFs are not supported.</p>
          
          <form onSubmit={handleUpload} className="space-y-5">
            <div className="flex items-center">
              <input
                id="retain"
                type="checkbox"
                checked={retainSource}
                onChange={(e) => setRetainSource(e.target.checked)}
                className="h-4 w-4 bg-inset border border-line rounded text-brand focus:ring-indigo-500"
              />
              <label htmlFor="retain" className="ml-2 block text-sm text-ink-muted">
                Retain original statement file on server
              </label>
            </div>

            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              className={`border-2 border-dashed rounded-lg p-8 transition cursor-pointer text-center relative ${
                dragActive
                  ? "border-indigo-500 bg-indigo-500/5"
                  : file
                    ? "border-emerald-500/40 bg-emerald-500/5"
                    : "border-line bg-inset/50 hover:bg-inset hover:border-line"
              }`}
            >
              <input
                type="file"
                accept=".pdf,.csv"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {file ? (
                <>
                  <FileUp className="mx-auto h-10 w-10 text-ok mb-2" />
                  <p className="text-sm font-semibold text-ok">{file.name}</p>
                  <p className="text-xs text-ink-subtle mt-1">{(file.size / 1024).toFixed(1)} KB · Click or drop to replace</p>
                </>
              ) : (
                <>
                  <Upload className="mx-auto h-12 w-12 text-ink-subtle mb-2" />
                  <p className="text-sm font-medium text-ink-muted">Click to select or drag PDF / CSV</p>
                  <p className="text-xs text-ink-subtle mt-1">PDF bank statement or CSV up to 10MB</p>
                </>
              )}
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-danger text-sm rounded-lg p-3 flex items-start gap-2">
                <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {warnings.length > 0 && !success && (
              <div className="bg-amber-500/10 border border-amber-500/20 text-warn text-sm rounded-lg p-3 flex items-start gap-2">
                <Info className="h-5 w-5 shrink-0 mt-0.5" />
                <div>
                  {warnings.map((w, i) => <p key={i} className="text-xs">{w}</p>)}
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !file}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:opacity-60 text-white font-medium rounded-lg shadow-sm transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Extracting transactions…
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4" />
                  Upload & Extract
                </>
              )}
            </button>
          </form>
        </div>
        
        {/* Success Panel */}
        <div className="h-fit">
          {success && (
            <div className="bg-gradient-to-br from-emerald-500/10 to-surface border border-emerald-500/20 text-ok rounded-xl p-6 animate-fade-in">
              <div className="flex items-start gap-4 mb-5">
                <CheckCircle2 className="h-8 w-8 text-ok shrink-0 mt-1" />
                <div>
                  <h3 className="text-lg font-bold text-ok mb-1">
                    {report?.status === "PARTIAL" ? "Statement partially processed" : "Statement imported"}
                  </h3>
                  <p className="text-ok/80 text-sm">
                    <span className="font-bold text-ok">{extractedCount}</span> new transactions stored; your profile was rebuilt from your full history.
                  </p>
                  {sourceType && (
                    <p className="text-xs text-ok/60 mt-1">
                      Source: <span className="font-mono">{sourceType}</span>
                    </p>
                  )}
                </div>
              </div>

              {warnings.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/20 text-warn text-xs rounded-lg p-3 mb-4">
                  {warnings.map((w, i) => <p key={i}>{w}</p>)}
                </div>
              )}

              {/* Import summary - counts only */}
              {report && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-5" id="import-summary">
                  {([
                    ["Transactions read", report.total_rows_detected],
                    ["Transactions imported", report.stored_transactions],
                    ["UPI transactions", report.upi_transactions ?? 0],
                    ["Debit transactions", report.debit_transactions],
                    ["Credit transactions", report.credit_transactions],
                    ["Reversal transactions", report.reversal_transactions ?? 0],
                    ["Counterparty names found", report.counterparty_names_found ?? 0],
                    ["Complete VPAs found", report.complete_vpas_found ?? report.upi_linked_transactions],
                    ["Partial VPA fragments found", report.partial_identifiers_found ?? 0],
                    ["Non-VPA identifiers found", report.non_vpa_identifiers_found ?? 0],
                    ["Rows with no counterparty identity", report.no_identity_rows ?? 0],
                    ["Rows rejected", report.rejected_transactions],
                    ["Rows partially parsed", report.partial_transactions],
                    ["Rows already imported", report.duplicate_transactions],
                    ["Valid rows", report.valid_transactions],
                    ["Payees updated", report.payee_relationships_updated],
                    ["Graph observations", report.graph_observations_updated],
                    ["Profile", report.profile_updated ? "Rebuilt" : "Not changed"],
                    ["Source", report.source_provider ?? (sourceType.includes("pdf") ? "PDF" : "CSV")],
                  ] as const).map(([label, value]) => (
                    <div key={label} className="bg-surface/60 rounded-lg p-3 border border-line/50">
                      <div className="text-xl font-bold text-ink">{value}</div>
                      <div className="text-xs text-ok/60">{label}</div>
                    </div>
                  ))}
                </div>
              )}
              {report && (report.unknown_direction > 0 || report.missing_timestamp > 0 ||
                report.malformed_upi_id > 0 || report.future_timestamp > 0 ||
                (report.partial_identifiers_found ?? 0) > 0 || (report.non_vpa_identifiers_found ?? 0) > 0 ||
                (report.no_identity_rows ?? 0) > 0 ||
                (report.cross_source_duplicates ?? 0) > 0 || (report.ambiguous_cross_source_kept ?? 0) > 0) && (
                <div className="bg-amber-500/10 border border-amber-500/20 text-warn text-xs rounded-lg p-3 mb-4 space-y-1">
                  {report.unknown_direction > 0 && <p>{report.unknown_direction} row(s): paid/received could not be read - kept out of spending and payee history.</p>}
                  {report.missing_timestamp > 0 && <p>{report.missing_timestamp} row(s): date could not be read - not used for timing features.</p>}
                  {report.future_timestamp > 0 && <p>{report.future_timestamp} row(s) are dated after today - not treated as past payments.</p>}
                  {report.malformed_upi_id > 0 && <p>{report.malformed_upi_id} row(s): the UPI ID was not a valid address.</p>}
                  {((report.partial_identifiers_found ?? 0) + (report.non_vpa_identifiers_found ?? 0)) > 0 && <p>Some bank statements truncate UPI identifiers. Partial identifiers are retained for behavioral history but are not used for exact VPA matching.</p>}
                  {(report.no_identity_rows ?? 0) > 0 && <p>{report.no_identity_rows} row(s) carry no counterparty name or identifier (e.g. interest, reversals, cash).</p>}
                  {(report.cross_source_duplicates ?? 0) > 0 && <p>{report.cross_source_duplicates} payment(s) were already imported from another statement and were not counted again.</p>}
                  {(report.ambiguous_cross_source_kept ?? 0) > 0 && <p>{report.ambiguous_cross_source_kept} row(s) looked similar to an earlier import but could not be confirmed as the same payment, so they were kept.</p>}
                </div>
              )}

              <button
                onClick={() => onNavigate?.("User Profile")}
                className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg shadow-sm transition flex items-center justify-center gap-2"
              >
                View Profile & Transactions
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
