import { useEffect, useMemo, useState } from "react";
import "./History.css";

const API = "http://127.0.0.1:8000";

export default function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [typeFilter, setTypeFilter] = useState("ALL");
  const [downloadingId, setDownloadingId] = useState(null);
  const [downloadError, setDownloadError] = useState("");
  const [deletingId, setDeletingId] = useState(null);
  const [deleteError, setDeleteError] = useState("");

  async function loadHistory() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API}/interviews?limit=100`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!Array.isArray(data)) throw new Error("Unexpected API response");
      setRecords(data);
    } catch (e) {
      setError(`Could not load history: ${e.message}. Check FastAPI.`);
    } finally {
      setLoading(false);
    }
  }

  async function downloadReport(recordId) {
    if (downloadingId !== null) return;
    setDownloadingId(recordId);
    setDownloadError("");
    try {
      const response = await fetch(`${API}/interviews/${recordId}/report`);
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { const body = await response.json(); if (body.detail) detail = body.detail; } catch { /* not JSON */ }
        throw new Error(detail);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `interview_report_${recordId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setDownloadError(`Could not download report #${recordId}: ${e.message}`);
    } finally {
      setDownloadingId(null);
    }
  }

  async function deleteRecord(record) {
    if (deletingId !== null || downloadingId !== null) return;
    const label = (record.record_type === "VIDEO_TEST" ? "video test" : "live interview");
    const confirmed = window.confirm(
      `Permanently delete ${label} #${record.id} (${record.interview_id})?\n\nThis cannot be undone.`
    );
    if (!confirmed) return;
    setDeletingId(record.id);
    setDeleteError("");
    try {
      const response = await fetch(`${API}/interviews/${record.id}`, { method: "DELETE" });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { const body = await response.json(); if (body.detail) detail = body.detail; } catch {}
        throw new Error(detail);
      }
      setRecords(previous => previous.filter(r => r.id !== record.id));
    } catch (e) {
      setDeleteError(`Could not delete record #${record.id}: ${e.message}`);
    } finally {
      setDeletingId(null);
    }
  }

  useEffect(() => { loadHistory(); }, []);

  const shown = useMemo(() => records.filter(r => {
    const matchesPrediction = filter === "ALL" || r.prediction === filter;
    const matchesType = typeFilter === "ALL" || (r.record_type || "LIVE_INTERVIEW") === typeFilter;
    const haystack = `${r.interview_id} ${r.candidate_name || ""} ${r.candidate_usn || ""} ${r.id}`.toLowerCase();
    return matchesPrediction && matchesType && haystack.includes(query.toLowerCase().trim());
  }), [records, query, filter, typeFilter]);

  return (
    <div className="history-page">
      <header className="history-top">
        <div><h1>Deepfake Guard</h1><p>All Detection History</p></div>
        <a href="/" className="history-back">← HR Dashboard</a>
      </header>
      <main className="history-content">
        <div className="history-heading">
          <div><h2>Saved Interviews &amp; Video Tests</h2><p>Up to 100 most recent saved results</p></div>
          <button onClick={loadHistory} disabled={loading}>Refresh</button>
        </div>
        <div className="history-filters">
          <input aria-label="Search records" placeholder="Search by interview ID, candidate, USN..." value={query} onChange={e => setQuery(e.target.value)} />
          <select aria-label="Filter prediction" value={filter} onChange={e => setFilter(e.target.value)}>
            <option value="ALL">All predictions</option><option value="REAL">REAL</option><option value="FAKE">FAKE</option>
          </select>
          <select aria-label="Filter record type" value={typeFilter} onChange={e => setTypeFilter(e.target.value)}>
            <option value="ALL">All types</option>
            <option value="LIVE_INTERVIEW">Live Interview</option>
            <option value="VIDEO_TEST">Video Test</option>
          </select>
        </div>
        {downloadError && <p role="alert" className="history-error">{downloadError}</p>}
        {deleteError && <p role="alert" className="history-error">{deleteError}</p>}
        {loading && <p role="status">Loading history...</p>}
        {error && <p role="alert" className="history-error">{error}</p>}
        {!loading && !error && (
          <>
            <p className="history-count">Showing {shown.length} of {records.length} saved records</p>
            {shown.length === 0 ? <p>No matching records found.</p> : (
              <div className="history-table-wrap"><table className="history-table">
                <thead><tr><th>Record</th><th>Type</th><th>Interview / Test ID</th><th>Candidate</th><th>USN</th><th>Prediction</th><th>Risk</th><th>Frames</th><th>Saved</th><th>PDF Report</th><th>Actions</th></tr></thead>
                <tbody>{shown.map(r => <tr key={r.id}>
                  <td>#{r.id}</td>
                  <td>{(r.record_type || "LIVE_INTERVIEW") === "VIDEO_TEST" ? "Video Test" : "Live Interview"}</td>
                  <td>{r.interview_id}</td><td>{r.candidate_name || "—"}</td><td>{r.candidate_usn || "—"}</td>
                  <td><span className={`history-status ${r.prediction === "FAKE" ? "history-fake" : "history-real"}`}>{r.prediction}</span></td>
                  <td>{Number(r.risk_score).toFixed(2)}%</td><td>{r.frames_analyzed}</td>
                  <td>{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                  <td><button type="button" className="history-download" disabled={downloadingId !== null || deletingId !== null} onClick={() => downloadReport(r.id)}>{downloadingId === r.id ? "Downloading..." : "↓ Download PDF"}</button></td>
                  <td><button type="button" className="history-delete" disabled={deletingId !== null || downloadingId !== null} onClick={() => deleteRecord(r)}>{deletingId === r.id ? "Deleting..." : "Delete"}</button></td>
                </tr>)}</tbody>
              </table></div>
            )}
          </>
        )}
        <p className="history-note">Completed live interviews and video tests are saved as results, not video recordings. Stop the video test after at least 30 analyzed frames to save it. Deleting a record permanently removes its saved details.</p>
      </main>
    </div>
  );
}
