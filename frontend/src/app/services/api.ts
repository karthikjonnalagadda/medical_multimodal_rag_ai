import type {
  AppSettings,
  EvidenceItem,
  FinalReportData,
  OcrResult,
  VisionResult,
} from "../types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000";

interface AnalyzeResponse {
  query_id: string;
  processing_time_ms: number;
  possible_conditions: Array<{
    name: string;
    confidence: string;
    icd_code?: string | null;
  }>;
  evidence: string[];
  references: string[];
  explanation: string;
  recommendation: string;
  disclaimer: string;
}

interface OcrApiResponse {
  filename?: string;
  raw_text: string;
  avg_confidence: number;
  metrics: Array<{
    name: string;
    value: string | number;
    unit?: string;
    in_normal_range?: boolean | null;
    reference_range?: string;
  }>;
  patient_info?: Record<string, string>;
}

interface VisionApiResponse {
  filename?: string;
  modality: string;
  image_quality: string;
  backend_name?: string;
  top_finding?: {
    label: string;
    confidence: number;
    description: string;
  } | null;
  all_findings: Array<{
    label: string;
    confidence: number;
    description: string;
    is_abnormal?: boolean;
  }>;
  top_differential?: Array<{
    label: string;
    confidence: number;
  }>;
}

function normalizeStatus(
  inNormalRange: boolean | null | undefined,
  value: string | number,
  referenceRange?: string,
): "normal" | "low" | "high" | "critical" {
  if (inNormalRange === true) {
    return "normal";
  }
  if (inNormalRange === false) {
    const numericValue = Number(value);
    if (referenceRange) {
      const [minRaw, maxRaw] = referenceRange.split("-").map((item) => item.replace(/[^\d.]/g, ""));
      const minValue = Number(minRaw);
      const maxValue = Number(maxRaw);
      if (!Number.isNaN(minValue) && numericValue < minValue) {
        return "low";
      }
      if (!Number.isNaN(maxValue) && numericValue > maxValue) {
        return numericValue > maxValue * 1.5 ? "critical" : "high";
      }
    }
    return "high";
  }
  return "normal";
}

function parsePercent(value: string): number {
  const numeric = Number(value.replace("%", "").trim());
  if (Number.isNaN(numeric)) {
    return 0;
  }
  return numeric > 1 ? numeric / 100 : numeric;
}

export async function fetchOcrResult(file: File): Promise<OcrResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/ocr`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`OCR request failed with ${response.status}`);
  }
  const payload = (await response.json()) as OcrApiResponse;
  return {
    rawText: payload.raw_text,
    confidence: payload.avg_confidence,
    metrics: payload.metrics.map((metric, index) => ({
      id: `metric-${index}`,
      name: metric.name,
      value: String(metric.value),
      unit: metric.unit || "",
      referenceRange: metric.reference_range || "",
      status: normalizeStatus(metric.in_normal_range, metric.value, metric.reference_range),
    })),
    documentType: payload.filename || "Medical report",
    extractedDate: payload.patient_info?.date || new Date().toISOString().slice(0, 10),
    patientId: payload.patient_info?.patient_id || payload.patient_info?.patient || "Unknown",
  };
}

export async function fetchVisionResult(file: File): Promise<VisionResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/image-analysis`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Image analysis request failed with ${response.status}`);
  }
  const payload = (await response.json()) as VisionApiResponse;
  const findings = payload.all_findings.map((finding, index) => ({
    id: `finding-${index}`,
    name: finding.label,
    confidence: finding.confidence,
    severity: finding.confidence >= 0.85 ? "high" : finding.confidence >= 0.6 ? "moderate" : "low",
    description: finding.description,
  })) as VisionResult["findings"];

  const topFinding = payload.top_finding?.label || findings[0]?.name || "No finding";
  const topConfidence = payload.top_finding?.confidence || findings[0]?.confidence || 0;

  return {
    topFinding,
    findings,
    confidence: topConfidence,
    modality: payload.modality,
    imageQuality: (
      payload.image_quality === "high"
        ? "excellent"
        : payload.image_quality === "medium"
        ? "good"
        : payload.image_quality === "low"
        ? "fair"
        : "poor"
    ),
    backendName: payload.backend_name || "Medical vision backend",
    gradcamOverlayUrl: null,
    originalImageUrl: "",
  };
}

export async function fetchAnalysisResult(args: {
  imageFile: File | null;
  pdfFile: File | null;
  symptoms: string;
  clinicalNotes: string;
  settings: AppSettings;
}): Promise<{
  finalReport: FinalReportData;
  evidence: EvidenceItem[];
}> {
  const formData = new FormData();
  if (args.pdfFile) {
    formData.append("lab_report", args.pdfFile);
  }
  if (args.imageFile) {
    formData.append("medical_image", args.imageFile);
  }
  if (args.symptoms.trim()) {
    formData.append("symptoms", args.symptoms);
  }
  if (args.clinicalNotes.trim()) {
    formData.append("patient_notes", args.clinicalNotes);
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Full analysis request failed with ${response.status}`);
  }
  const payload = (await response.json()) as AnalyzeResponse;

  const evidence = payload.references.map((reference, index) => ({
    id: `evidence-${index}`,
    source: reference.includes("PubMed")
      ? "PubMed"
      : reference.includes("ICD")
      ? "ICD-10"
      : reference.includes("WHO") || reference.includes("Guideline")
      ? "Guidelines"
      : "Local KB",
    title: reference,
    snippet: payload.evidence[index] || payload.explanation,
    relevanceScore: Math.max(0.55, 0.95 - index * 0.08),
    metadata: {},
  })) as EvidenceItem[];

  const finalReport: FinalReportData = {
    possibleFindings: payload.possible_conditions.slice(0, 3).map((condition, index) => ({
      rank: index + 1,
      condition: condition.name,
      probability: parsePercent(condition.confidence),
      supportingEvidence: payload.evidence.slice(index, index + 2),
      icdCode: condition.icd_code || "N/A",
    })),
    explanation: payload.explanation,
    evidence,
    recommendation: payload.recommendation,
    generatedAt: new Date().toISOString(),
    modelUsed: args.settings.modelBackend,
    sessionId: payload.query_id,
  };

  return { finalReport, evidence };
}

export async function fetchKnowledgeStats(): Promise<{ totalDocuments: number; backend: string }> {
  const response = await fetch(`${API_BASE_URL}/knowledge/stats`);
  if (!response.ok) {
    throw new Error(`Knowledge stats request failed with ${response.status}`);
  }
  const payload = await response.json();
  return {
    totalDocuments: payload.total_documents,
    backend: payload.backend,
  };
}

// Chat API endpoints

interface ChatRequestPayload {
  message: string;
}

interface ChatSource {
  text: string;
  metadata?: Record<string, string>;
}

interface ChatResponsePayload {
  answer: string;
  sources: ChatSource[];
  query: string;
  retrieval_time: number;
  llm_time: number;
  total_time: number;
  structured?: {
    summary?: string;
    sections?: Array<{ title: string; content: string }>;
    full_text?: string;
  };
}

interface ChatStatsPayload {
  total_documents: number;
  embedding_model: string;
  vector_store_backend: string;
  status: string;
}

export async function fetchChatResponse(message: string): Promise<ChatResponsePayload> {
  const payload: ChatRequestPayload = { message };
  
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed with status ${response.status}`);
  }

  const data = (await response.json()) as ChatResponsePayload;
  return data;
}

export async function fetchChatStats(): Promise<ChatStatsPayload> {
  const response = await fetch(`${API_BASE_URL}/chat/stats`);

  if (!response.ok) {
    throw new Error(`Chat stats request failed with status ${response.status}`);
  }

  const data = (await response.json()) as ChatStatsPayload;
  return data;
}
