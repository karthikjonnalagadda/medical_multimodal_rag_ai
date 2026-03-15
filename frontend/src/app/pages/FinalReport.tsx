import React, { useCallback } from "react";
import { useAppContext } from "../context/AppContext";
import { StatusBadge } from "../components/ui/StatusBadge";
import { ConfidenceBar } from "../components/ui/ConfidenceBar";
import {
  ClipboardList,
  Download,
  Printer,
  FileJson,
  FileText,
  AlertCircle,
  Lightbulb,
  BookOpen,
  ArrowRight,
  Shield,
  Clock,
  Cpu,
  Hash,
} from "lucide-react";

export function FinalReport() {
  const { finalReport } = useAppContext();

  const exportJSON = useCallback(() => {
    if (!finalReport) return;
    const blob = new Blob([JSON.stringify(finalReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `medical-report-${finalReport.sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [finalReport]);

  const exportText = useCallback(() => {
    if (!finalReport) return;
    const text = [
      "MULTIMODAL MEDICAL AI REPORT",
      `Session: ${finalReport.sessionId}`,
      `Generated: ${new Date(finalReport.generatedAt).toLocaleString()}`,
      `Model: ${finalReport.modelUsed}`,
      "",
      "═══ DIFFERENTIAL DIAGNOSIS ═══",
      ...finalReport.possibleFindings.map(
        (f) =>
          `\n#${f.rank} ${f.condition} (${Math.round(f.probability * 100)}%) [${f.icdCode}]\n` +
          f.supportingEvidence.map((e) => `  - ${e}`).join("\n")
      ),
      "",
      "═══ EXPLANATION ═══",
      finalReport.explanation,
      "",
      "═══ RECOMMENDATION ═══",
      finalReport.recommendation,
    ].join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `medical-report-${finalReport.sessionId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [finalReport]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  if (!finalReport) {
    return (
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <h1 className="text-foreground flex items-center gap-2 mb-2">
          <ClipboardList className="w-5 h-5 text-primary" />
          Final Report
        </h1>
        <div className="text-center py-20 bg-card rounded-xl border border-dashed border-border mt-6">
          <ClipboardList className="w-12 h-12 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-[14px] text-muted-foreground">No report generated yet</p>
          <p className="text-[12px] text-muted-foreground/70 mt-1">
            Complete the analysis pipeline to generate a structured report
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-foreground flex items-center gap-2">
            <ClipboardList className="w-5 h-5 text-primary" />
            Diagnostic Report
          </h1>
          <p className="text-[13px] text-muted-foreground mt-1">
            Structured findings, clinical reasoning, and recommendations.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={exportJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-[12px] text-foreground hover:bg-secondary transition-colors"
          >
            <FileJson className="w-3.5 h-3.5" />
            Export JSON
          </button>
          <button
            onClick={exportText}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-[12px] text-foreground hover:bg-secondary transition-colors"
          >
            <FileText className="w-3.5 h-3.5" />
            Export Text
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-[12px] hover:bg-primary/90 transition-colors"
          >
            <Printer className="w-3.5 h-3.5" />
            Print
          </button>
        </div>
      </div>

      {/* Report metadata */}
      <div className="flex flex-wrap items-center gap-3 text-[11px]">
        <span className="flex items-center gap-1 text-muted-foreground">
          <Clock className="w-3 h-3" />
          {new Date(finalReport.generatedAt).toLocaleString()}
        </span>
        <span className="flex items-center gap-1 text-muted-foreground">
          <Cpu className="w-3 h-3" />
          {finalReport.modelUsed}
        </span>
        <span className="flex items-center gap-1 text-muted-foreground">
          <Hash className="w-3 h-3" />
          {finalReport.sessionId}
        </span>
      </div>

      {/* Section 1: Differential Diagnosis */}
      <section className="bg-card rounded-xl border border-border overflow-hidden">
        <div className="px-6 py-4 border-b border-border bg-secondary/30">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-primary" />
            <h2 className="text-foreground">Possible Findings</h2>
          </div>
          <p className="text-[12px] text-muted-foreground mt-0.5">Top differential diagnoses ranked by probability</p>
        </div>
        <div className="divide-y divide-border">
          {finalReport.possibleFindings.map((dx) => (
            <div key={dx.rank} className="px-6 py-5">
              <div className="flex items-start gap-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
                  dx.rank === 1 ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"
                }`}>
                  <span className="text-[14px]">#{dx.rank}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 flex-wrap mb-2">
                    <h3 className="text-foreground">{dx.condition}</h3>
                    <StatusBadge variant="neutral">{dx.icdCode}</StatusBadge>
                  </div>
                  <ConfidenceBar value={dx.probability} className="mb-3 max-w-sm" />
                  <div className="space-y-1">
                    {dx.supportingEvidence.map((ev, i) => (
                      <div key={i} className="flex items-start gap-2 text-[12px] text-muted-foreground">
                        <span className="text-primary mt-0.5">-</span>
                        <span>{ev}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Section 2: Explanation */}
      <section className="bg-card rounded-xl border border-border overflow-hidden">
        <div className="px-6 py-4 border-b border-border bg-secondary/30">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-amber-500" />
            <h2 className="text-foreground">Clinical Explanation</h2>
          </div>
        </div>
        <div className="px-6 py-5">
          {finalReport.explanation.split("\n\n").map((paragraph, i) => (
            <p key={i} className="text-[13px] text-foreground/80 leading-relaxed mb-4 last:mb-0">
              {paragraph}
            </p>
          ))}
        </div>
      </section>

      {/* Section 3: Evidence */}
      <section className="bg-card rounded-xl border border-border overflow-hidden">
        <div className="px-6 py-4 border-b border-border bg-secondary/30">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-blue-500" />
            <h2 className="text-foreground">Supporting Evidence</h2>
          </div>
          <p className="text-[12px] text-muted-foreground mt-0.5">{finalReport.evidence.length} key references cited</p>
        </div>
        <div className="divide-y divide-border">
          {finalReport.evidence.map((ev) => (
            <div key={ev.id} className="px-6 py-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <StatusBadge variant="info">{ev.source}</StatusBadge>
                    {ev.metadata.year && (
                      <span className="text-[11px] text-muted-foreground">{ev.metadata.year}</span>
                    )}
                  </div>
                  <p className="text-[13px] text-foreground">{ev.title}</p>
                  {ev.metadata.authors && (
                    <p className="text-[11px] text-muted-foreground mt-0.5">{ev.metadata.authors}</p>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <span className="text-[14px] text-foreground">{Math.round(ev.relevanceScore * 100)}%</span>
                  <p className="text-[10px] text-muted-foreground">match</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Section 4: Recommendation */}
      <section className="bg-card rounded-xl border border-border overflow-hidden">
        <div className="px-6 py-4 border-b border-border bg-primary/5">
          <div className="flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-primary" />
            <h2 className="text-foreground">Recommendations</h2>
          </div>
        </div>
        <div className="px-6 py-5">
          <div className="space-y-2">
            {finalReport.recommendation.split("\n").map((line, i) => (
              <div key={i} className="flex items-start gap-3 text-[13px]">
                <span className="w-5 h-5 rounded-full bg-primary/10 text-primary flex items-center justify-center text-[10px] shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <p className="text-foreground/80">{line.replace(/^\d+\.\s*/, "")}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200">
        <Shield className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <p className="text-[12px] text-amber-800">
            This report is generated by an AI system for clinical decision support purposes only.
            All findings, diagnoses, and recommendations must be independently verified by a qualified
            healthcare professional. This system does not replace clinical judgment.
          </p>
        </div>
      </div>
    </div>
  );
}
