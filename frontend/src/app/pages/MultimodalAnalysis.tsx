import React, { useCallback, useState } from "react";
import { useAppContext } from "../context/AppContext";
import { PipelineProgress } from "../components/ui/PipelineProgress";
import { ConfidenceBar } from "../components/ui/ConfidenceBar";
import { StatusBadge } from "../components/ui/StatusBadge";
import {
  Upload,
  FileImage,
  FileText,
  Play,
  Stethoscope,
  Target,
  BookOpen,
  ArrowRight,
  Sparkles,
  X,
} from "lucide-react";

export function MultimodalAnalysis() {
  const {
    uploadedImage, setUploadedImage,
    uploadedPdf, setUploadedPdf,
    setImageFile, setPdfFile,
    symptoms, setSymptoms,
    clinicalNotes, setClinicalNotes,
    isAnalysisRunning, analysisComplete,
    pipelineSteps, visionResult, ocrResult,
    evidence, finalReport,
    lastError,
    runAnalysis,
  } = useAppContext();

  const [dragOverImage, setDragOverImage] = useState(false);
  const [dragOverPdf, setDragOverPdf] = useState(false);

  const handleImageDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOverImage(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => setUploadedImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, [setImageFile, setUploadedImage]);

  const handlePdfDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOverPdf(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setPdfFile(file);
      setUploadedPdf(file.name);
    }
  }, [setPdfFile, setUploadedPdf]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => setUploadedImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handlePdfSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPdfFile(file);
      setUploadedPdf(file.name);
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-foreground flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            Multimodal Analysis
          </h1>
          <p className="text-[13px] text-muted-foreground mt-1">
            Upload medical images and documents, enter clinical context, and run the full diagnostic pipeline.
          </p>
        </div>
        {analysisComplete && (
          <StatusBadge variant="success" dot>Analysis Complete</StatusBadge>
        )}
      </div>

      {/* Upload section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Image upload */}
        <div
          className={`relative border-2 border-dashed rounded-xl p-6 transition-colors cursor-pointer ${
            dragOverImage ? "border-primary bg-primary/5" : uploadedImage ? "border-emerald-300 bg-emerald-50/30" : "border-border hover:border-primary/40"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOverImage(true); }}
          onDragLeave={() => setDragOverImage(false)}
          onDrop={handleImageDrop}
          onClick={() => !uploadedImage && document.getElementById("img-upload")?.click()}
        >
          <input id="img-upload" type="file" accept="image/*" className="hidden" onChange={handleImageSelect} />
          {uploadedImage ? (
            <div className="flex items-start gap-4">
              <img src={uploadedImage} alt="Uploaded" className="w-24 h-24 object-cover rounded-lg" />
              <div className="flex-1 min-w-0">
                <p className="text-[13px] text-foreground">Medical image uploaded</p>
                <p className="text-[11px] text-muted-foreground mt-1">Ready for vision analysis</p>
                <StatusBadge variant="success" dot>Uploaded</StatusBadge>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setUploadedImage(null); setImageFile(null); }}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="text-center py-4">
              <FileImage className="w-10 h-10 mx-auto text-muted-foreground/40 mb-3" />
              <p className="text-[13px] text-foreground">Drop medical image here</p>
              <p className="text-[11px] text-muted-foreground mt-1">X-ray, CT, MRI — PNG, JPEG, DICOM</p>
            </div>
          )}
        </div>

        {/* PDF upload */}
        <div
          className={`relative border-2 border-dashed rounded-xl p-6 transition-colors cursor-pointer ${
            dragOverPdf ? "border-primary bg-primary/5" : uploadedPdf ? "border-emerald-300 bg-emerald-50/30" : "border-border hover:border-primary/40"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOverPdf(true); }}
          onDragLeave={() => setDragOverPdf(false)}
          onDrop={handlePdfDrop}
          onClick={() => !uploadedPdf && document.getElementById("pdf-upload")?.click()}
        >
          <input id="pdf-upload" type="file" accept=".pdf,.doc,.docx" className="hidden" onChange={handlePdfSelect} />
          {uploadedPdf ? (
            <div className="flex items-start gap-4">
              <div className="w-24 h-24 rounded-lg bg-blue-50 flex items-center justify-center">
                <FileText className="w-10 h-10 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] text-foreground truncate">{uploadedPdf}</p>
                <p className="text-[11px] text-muted-foreground mt-1">Ready for OCR extraction</p>
                <StatusBadge variant="success" dot>Uploaded</StatusBadge>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setUploadedPdf(null); setPdfFile(null); }}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="text-center py-4">
              <Upload className="w-10 h-10 mx-auto text-muted-foreground/40 mb-3" />
              <p className="text-[13px] text-foreground">Drop lab report or PDF</p>
              <p className="text-[11px] text-muted-foreground mt-1">PDF, DOC — Lab results, discharge summaries</p>
            </div>
          )}
        </div>
      </div>

      {/* Clinical context */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-card rounded-xl border border-border p-5">
          <label className="text-[12px] text-muted-foreground mb-2 block">Symptoms & Chief Complaint</label>
          <textarea
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            placeholder="e.g., Productive cough x 5 days, fever 38.9C, right-sided pleuritic chest pain, dyspnea on exertion..."
            className="w-full h-28 px-3 py-2.5 rounded-lg border border-border bg-input-background text-[13px] text-foreground placeholder:text-muted-foreground/50 resize-none"
          />
        </div>
        <div className="bg-card rounded-xl border border-border p-5">
          <label className="text-[12px] text-muted-foreground mb-2 block">Clinical Notes</label>
          <textarea
            value={clinicalNotes}
            onChange={(e) => setClinicalNotes(e.target.value)}
            placeholder="e.g., 62M, smoker, HTN, DM2. No recent travel. No known TB contacts. Completed COVID vaccination..."
            className="w-full h-28 px-3 py-2.5 rounded-lg border border-border bg-input-background text-[13px] text-foreground placeholder:text-muted-foreground/50 resize-none"
          />
        </div>
      </div>

      {/* Run analysis */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => {
            void runAnalysis();
          }}
          disabled={isAnalysisRunning}
          className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-[13px]"
        >
          <Play className="w-4 h-4" />
          {isAnalysisRunning ? "Running Pipeline..." : "Run Full Analysis"}
        </button>
        {(isAnalysisRunning || analysisComplete) && (
          <PipelineProgress steps={pipelineSteps} />
        )}
      </div>

      {lastError && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[12px] text-amber-900">
          {lastError}
        </div>
      )}

      {/* Results summary */}
      {analysisComplete && visionResult && finalReport && (
        <div className="space-y-5">
          <h2 className="text-foreground flex items-center gap-2">
            <Target className="w-4 h-4 text-primary" />
            Analysis Summary
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Top finding */}
            <div className="bg-card rounded-xl border border-border p-5">
              <div className="flex items-center gap-2 mb-3">
                <Stethoscope className="w-4 h-4 text-primary" />
                <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Top Finding</span>
              </div>
              <p className="text-[14px] text-foreground">{visionResult.topFinding}</p>
              <div className="mt-2">
                <ConfidenceBar value={visionResult.confidence} size="sm" />
              </div>
            </div>

            {/* Confidence */}
            <div className="bg-card rounded-xl border border-border p-5">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-4 h-4 text-emerald-500" />
                <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Confidence</span>
              </div>
              <p className="text-[28px] text-foreground">{Math.round(visionResult.confidence * 100)}%</p>
              <StatusBadge variant={visionResult.confidence > 0.8 ? "success" : "warning"}>
                {visionResult.confidence > 0.8 ? "High confidence" : "Moderate"}
              </StatusBadge>
            </div>

            {/* Evidence count */}
            <div className="bg-card rounded-xl border border-border p-5">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen className="w-4 h-4 text-blue-500" />
                <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Evidence</span>
              </div>
              <p className="text-[28px] text-foreground">{evidence.length}</p>
              <p className="text-[12px] text-muted-foreground">sources retrieved</p>
            </div>

            {/* Recommendation */}
            <div className="bg-card rounded-xl border border-border p-5">
              <div className="flex items-center gap-2 mb-3">
                <ArrowRight className="w-4 h-4 text-amber-500" />
                <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Next Action</span>
              </div>
              <p className="text-[13px] text-foreground leading-relaxed">
                {finalReport.recommendation.split("\n")[0]}
              </p>
            </div>
          </div>

          {/* Differential diagnosis preview */}
          <div className="bg-card rounded-xl border border-border p-5">
            <h3 className="text-foreground mb-4">Differential Diagnosis</h3>
            <div className="space-y-3">
              {finalReport.possibleFindings.map((dx) => (
                <div key={dx.rank} className="flex items-center gap-4">
                  <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-[12px] shrink-0">
                    {dx.rank}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-[13px] text-foreground truncate">{dx.condition}</p>
                      <StatusBadge variant="neutral">{dx.icdCode}</StatusBadge>
                    </div>
                    <ConfidenceBar value={dx.probability} size="sm" className="mt-1" />
                  </div>
                  <span className="text-[13px] text-muted-foreground">{Math.round(dx.probability * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!isAnalysisRunning && !analysisComplete && (
        <div className="text-center py-12 bg-card rounded-xl border border-dashed border-border">
          <Stethoscope className="w-12 h-12 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-[14px] text-muted-foreground">No analysis results yet</p>
          <p className="text-[12px] text-muted-foreground/70 mt-1">
            Upload files and click "Run Full Analysis" to begin
          </p>
        </div>
      )}
    </div>
  );
}
