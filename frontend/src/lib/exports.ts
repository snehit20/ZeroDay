// ─── Export Utilities ─────────────────────────────────────────────
// Client-side download triggers for reports, datasets, and models

export function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  triggerDownload(blob, filename);
}

export function downloadCSV(headers: string[], rows: string[][], filename: string) {
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  triggerDownload(blob, filename);
}

export function downloadPDF(data: unknown, filename: string) {
  // Generates a simple text-based report as PDF-compatible content
  const report = generateTextReport(data);
  const blob = new Blob([report], { type: "text/plain" });
  triggerDownload(blob, filename.replace(".pdf", ".txt"));
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function generateTextReport(data: any): string {
  const lines: string[] = [
    "═══════════════════════════════════════════════════════════════",
    "  AEGIS — AI Bias Governance Report",
    "═══════════════════════════════════════════════════════════════",
    "",
    `Date: ${new Date().toISOString()}`,
    `Mode: ${data?.metadata?.mode || "full_pipeline"}`,
    `Strategy: ${data?.metadata?.strategy_used || "N/A"}`,
    `Elapsed: ${data?.metadata?.elapsed_seconds || 0}s`,
    "",
    "─── Summary ───",
    data?.explanations?.summary || "",
    "",
    "─── Bias Explanation ───",
    data?.explanations?.bias_explanation || "",
    "",
    "─── Strategy Justification ───",
    data?.explanations?.strategy_justification || "",
    "",
    "─── Tradeoff Analysis ───",
    data?.explanations?.tradeoff_analysis || "",
    "",
    "─── Recommendation ───",
    data?.explanations?.recommendation || "",
    "",
    "═══════════════════════════════════════════════════════════════",
  ];
  return lines.join("\n");
}
