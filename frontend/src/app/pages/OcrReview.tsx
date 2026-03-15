import React, { useState, useMemo } from "react";
import { useAppContext } from "../context/AppContext";
import { StatusBadge } from "../components/ui/StatusBadge";
import {
  FileText,
  Search,
  AlertTriangle,
  CheckCircle2,
  Filter,
  Calendar,
  User,
  FileType,
  Gauge,
} from "lucide-react";

export function OcrReview() {
  const { ocrResult } = useAppContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [showAbnormalOnly, setShowAbnormalOnly] = useState(false);
  const [activeTab, setActiveTab] = useState<"table" | "raw">("table");

  const rawTextForMatch = useMemo(() => {
    const raw = (ocrResult?.rawText || "").toLowerCase();
    // Normalize common OCR number formatting so "15,000" matches "15000" and vice-versa.
    const withoutThousandsSeparators = raw.replace(/(\d),(?=\d{3}\b)/g, "$1");
    const normalizedWhitespace = withoutThousandsSeparators.replace(/\s+/g, " ");
    return normalizedWhitespace;
  }, [ocrResult]);

  function toTitleCase(input: string): string {
    return input.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function isMetricValueInRawText(value: string, rawText: string): boolean {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed) return false;

    const candidates = new Set<string>();
    candidates.add(trimmed);

    const withoutThousandsSeparators = trimmed.replace(/(\d),(?=\d{3}\b)/g, "$1");
    if (withoutThousandsSeparators && withoutThousandsSeparators !== trimmed) {
      candidates.add(withoutThousandsSeparators);
    }
    const normalizedWhitespace = trimmed.replace(/\s+/g, " ");
    if (normalizedWhitespace && normalizedWhitespace !== trimmed) {
      candidates.add(normalizedWhitespace);
    }

    // Also try matching values where OCR uses comma as decimal separator.
    if (trimmed.includes(".")) {
      candidates.add(trimmed.replaceAll(".", ","));
    }
    if (trimmed.includes(",")) {
      candidates.add(trimmed.replaceAll(",", "."));
    }

    // If it looks numeric, try a digits-only variant (handles spacing, units stuck to numbers, etc).
    const digitsOnly = trimmed.replace(/[^\d.]/g, "");
    if (digitsOnly && digitsOnly !== trimmed) {
      candidates.add(digitsOnly);
    }

    for (const needle of candidates) {
      if (needle && rawText.includes(needle)) {
        return true;
      }
    }
    return false;
  }

  const filteredMetrics = useMemo(() => {
    if (!ocrResult) return [];
    let metrics = ocrResult.metrics;
    if (showAbnormalOnly) {
      metrics = metrics.filter((m) => m.status !== "normal");
    }
    if (searchTerm) {
      metrics = metrics.filter(
        (m) =>
          m.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          m.value.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    return metrics;
  }, [ocrResult, searchTerm, showAbnormalOnly]);

  const abnormalCount = ocrResult?.metrics.filter((m) => m.status !== "normal").length ?? 0;

  if (!ocrResult) {
    return (
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <h1 className="text-foreground flex items-center gap-2 mb-2">
          <FileText className="w-5 h-5 text-primary" />
          OCR Review
        </h1>
        <div className="text-center py-20 bg-card rounded-xl border border-dashed border-border mt-6">
          <FileText className="w-12 h-12 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-[14px] text-muted-foreground">No OCR results available</p>
          <p className="text-[12px] text-muted-foreground/70 mt-1">
            Upload a lab report and run the analysis pipeline
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-foreground flex items-center gap-2">
          <FileText className="w-5 h-5 text-primary" />
          OCR Review
        </h1>
        <p className="text-[13px] text-muted-foreground mt-1">
          Extracted text and structured lab values from the uploaded document.
        </p>
      </div>

      {/* Meta cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-3">
        {[
          { icon: Gauge, label: "OCR Confidence", value: `${Math.round(ocrResult.confidence * 100)}%`, variant: "success" as const },
          { icon: FileType, label: "Document Type", value: ocrResult.documentType.split(" - ")[0], variant: "info" as const },
          { icon: Calendar, label: "Report Date", value: ocrResult.extractedDate, variant: "neutral" as const },
          { icon: User, label: "Patient ID", value: ocrResult.patientId, variant: "neutral" as const },
          { icon: AlertTriangle, label: "Abnormal Values", value: `${abnormalCount} flagged`, variant: abnormalCount > 3 ? "error" as const : "warning" as const },
        ].map((item) => (
          <div key={item.label} className="bg-card rounded-xl border border-border p-4">
            <div className="flex items-center gap-1.5 mb-2">
              <item.icon className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground">{item.label}</span>
            </div>
            <StatusBadge variant={item.variant}>{item.value}</StatusBadge>
          </div>
        ))}
      </div>

      {/* Tabs and search */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-1 bg-secondary rounded-lg p-0.5">
          {(["table", "raw"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded-md text-[12px] transition-colors ${
                activeTab === tab
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab === "table" ? "Structured Values" : "Raw Text"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search values..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-lg border border-border bg-input-background text-[12px] text-foreground w-48"
            />
          </div>
          <button
            onClick={() => setShowAbnormalOnly(!showAbnormalOnly)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] transition-colors border ${
              showAbnormalOnly
                ? "bg-amber-50 border-amber-200 text-amber-700"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <Filter className="w-3.5 h-3.5" />
            Abnormal only
          </button>
        </div>
      </div>

      {/* Content */}
      {activeTab === "table" ? (
        <div className="bg-card rounded-xl border border-border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="bg-secondary/50">
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Test</th>
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Value</th>
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Unit</th>
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Reference</th>
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Matched</th>
                  <th className="text-left px-5 py-3 text-[11px] text-muted-foreground uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredMetrics.map((metric) => {
                  const matched = isMetricValueInRawText(metric.value, rawTextForMatch);
                  const displayName = toTitleCase(metric.name);
                  return (
                  <tr
                    key={metric.id}
                    className={`border-t border-border transition-colors ${
                      metric.status === "critical"
                        ? "bg-red-50/50"
                        : metric.status === "high"
                        ? "bg-amber-50/30"
                        : metric.status === "low"
                        ? "bg-blue-50/30"
                        : "hover:bg-secondary/30"
                    }`}
                  >
                    <td className="px-5 py-3 text-foreground">{displayName}</td>
                    <td className={`px-5 py-3 ${
                      metric.status !== "normal" ? "text-foreground" : "text-foreground"
                    }`}>
                      <span className={metric.status !== "normal" ? "px-1.5 py-0.5 rounded bg-amber-100 text-amber-800" : ""}>
                        {metric.value}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">{metric.unit}</td>
                    <td className="px-5 py-3 text-muted-foreground">{metric.referenceRange}</td>
                    <td className="px-5 py-3">
                      {matched ? (
                        <span className="flex items-center gap-1 text-emerald-600">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Yes
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-amber-700">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          No
                        </span>
                      )}
                    </td>
                    <td className="px-5 py-3">
                      {metric.status === "normal" ? (
                        <span className="flex items-center gap-1 text-emerald-600">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Normal
                        </span>
                      ) : (
                        <StatusBadge
                          variant={metric.status === "critical" ? "critical" : metric.status === "high" ? "warning" : "info"}
                          dot
                        >
                          {metric.status === "critical" ? "Critical" : metric.status === "high" ? "High" : "Low"}
                        </StatusBadge>
                      )}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {filteredMetrics.length === 0 && (
            <div className="text-center py-8 text-[13px] text-muted-foreground">
              No matching results found
            </div>
          )}
        </div>
      ) : (
        <div className="bg-card rounded-xl border border-border p-5">
          <div className="bg-slate-900 rounded-lg p-5 overflow-auto max-h-[600px]">
            <pre className="text-[12px] text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
              {ocrResult.rawText}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
