import React, { createContext, useContext, useState, type ReactNode } from "react";
import type { AppSettings, VisionResult, OcrResult, EvidenceItem, FinalReportData, PipelineStep } from "../types";
import { mockVisionResult, mockOcrResult, mockEvidence, mockFinalReport, initialPipelineSteps } from "../services/mockData";
import { fetchAnalysisResult, fetchOcrResult, fetchVisionResult } from "../services/api";

interface AppState {
  settings: AppSettings;
  setSettings: (s: AppSettings) => void;
  visionResult: VisionResult | null;
  setVisionResult: (v: VisionResult | null) => void;
  ocrResult: OcrResult | null;
  setOcrResult: (o: OcrResult | null) => void;
  evidence: EvidenceItem[];
  setEvidence: (e: EvidenceItem[]) => void;
  finalReport: FinalReportData | null;
  setFinalReport: (r: FinalReportData | null) => void;
  pipelineSteps: PipelineStep[];
  setPipelineSteps: (s: PipelineStep[]) => void;
  uploadedImage: string | null;
  setUploadedImage: (u: string | null) => void;
  uploadedPdf: string | null;
  setUploadedPdf: (u: string | null) => void;
  imageFile: File | null;
  setImageFile: (file: File | null) => void;
  pdfFile: File | null;
  setPdfFile: (file: File | null) => void;
  symptoms: string;
  setSymptoms: (s: string) => void;
  clinicalNotes: string;
  setClinicalNotes: (s: string) => void;
  isAnalysisRunning: boolean;
  setIsAnalysisRunning: (b: boolean) => void;
  analysisComplete: boolean;
  setAnalysisComplete: (b: boolean) => void;
  lastError: string | null;
  runAnalysis: () => Promise<void>;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>({
    confidenceThreshold: 0.5,
    topK: 5,
    modelBackend: "Groq + Medical RAG",
    modalityFilter: "all",
  });
  const [visionResult, setVisionResult] = useState<VisionResult | null>(null);
  const [ocrResult, setOcrResult] = useState<OcrResult | null>(null);
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [finalReport, setFinalReport] = useState<FinalReportData | null>(null);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(initialPipelineSteps);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedPdf, setUploadedPdf] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [symptoms, setSymptoms] = useState("");
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const runAnalysis = async () => {
    setIsAnalysisRunning(true);
    setAnalysisComplete(false);
    setLastError(null);
    setVisionResult(null);
    setOcrResult(null);
    setEvidence([]);
    setFinalReport(null);

    const steps = initialPipelineSteps.map((step) => ({ ...step, status: "pending" as const, duration: undefined }));
    setPipelineSteps(steps);

    const updateStep = (id: string, status: PipelineStep["status"]) => {
      setPipelineSteps((prev) =>
        prev.map((step) => (step.id === id ? { ...step, status } : step))
      );
    };

    try {
      if (!imageFile && !pdfFile && !symptoms.trim() && !clinicalNotes.trim()) {
        throw new Error("Please upload an image or report, or enter symptoms/clinical notes before running analysis");
      }

      if (pdfFile) {
        updateStep("ocr", "running");
        const ocr = await fetchOcrResult(pdfFile);
        setOcrResult(ocr);
        updateStep("ocr", "completed");
      } else {
        updateStep("ocr", "completed");
      }

      if (imageFile) {
        updateStep("vision", "running");
        const vision = await fetchVisionResult(imageFile);
        vision.originalImageUrl = uploadedImage || "";
        setVisionResult(vision);
        updateStep("vision", "completed");
      } else {
        updateStep("vision", "completed");
      }

      updateStep("retrieval", "running");
      updateStep("reasoning", "running");
      const analysis = await fetchAnalysisResult({
        imageFile,
        pdfFile,
        symptoms,
        clinicalNotes,
        settings,
      });
      setEvidence(analysis.evidence);
      setFinalReport(analysis.finalReport);
      updateStep("retrieval", "completed");
      updateStep("reasoning", "completed");
      setAnalysisComplete(true);
    } catch (error) {
      console.error(error);
      setLastError(
        error instanceof Error
          ? `${error.message}. Falling back to demo data.`
          : "Backend unavailable. Falling back to demo data.",
      );
      setOcrResult(pdfFile ? mockOcrResult : null);
      setVisionResult(imageFile ? { ...mockVisionResult, originalImageUrl: uploadedImage || "" } : null);
      setEvidence(mockEvidence);
      setFinalReport(mockFinalReport);
      setPipelineSteps((prev) =>
        prev.map((step) => ({
          ...step,
          status: step.id === "reasoning" ? "error" : "completed",
        }))
      );
      setAnalysisComplete(true);
    } finally {
      setIsAnalysisRunning(false);
    }
  };

  return (
    <AppContext.Provider
      value={{
        settings, setSettings,
        visionResult, setVisionResult,
        ocrResult, setOcrResult,
        evidence, setEvidence,
        finalReport, setFinalReport,
        pipelineSteps, setPipelineSteps,
        uploadedImage, setUploadedImage,
        uploadedPdf, setUploadedPdf,
        imageFile, setImageFile,
        pdfFile, setPdfFile,
        symptoms, setSymptoms,
        clinicalNotes, setClinicalNotes,
        isAnalysisRunning, setIsAnalysisRunning,
        analysisComplete, setAnalysisComplete,
        lastError,
        runAnalysis,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}
