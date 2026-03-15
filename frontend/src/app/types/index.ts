export interface Finding {
  id: string;
  name: string;
  confidence: number;
  severity: "low" | "moderate" | "high" | "critical";
  description: string;
}

export interface VisionResult {
  topFinding: string;
  findings: Finding[];
  confidence: number;
  modality: string;
  imageQuality: "excellent" | "good" | "fair" | "poor";
  backendName: string;
  gradcamOverlayUrl: string | null;
  originalImageUrl: string;
}

export interface LabMetric {
  id: string;
  name: string;
  value: string;
  unit: string;
  referenceRange: string;
  status: "normal" | "low" | "high" | "critical";
}

export interface OcrResult {
  rawText: string;
  confidence: number;
  metrics: LabMetric[];
  documentType: string;
  extractedDate: string;
  patientId: string;
}

export interface EvidenceItem {
  id: string;
  source: "PubMed" | "Guidelines" | "ICD-10" | "Local KB" | "UpToDate";
  title: string;
  snippet: string;
  relevanceScore: number;
  metadata: {
    authors?: string;
    year?: string;
    journal?: string;
    specialty?: string;
    doi?: string;
  };
}

export interface DifferentialDiagnosis {
  rank: number;
  condition: string;
  probability: number;
  supportingEvidence: string[];
  icdCode: string;
}

export interface FinalReportData {
  possibleFindings: DifferentialDiagnosis[];
  explanation: string;
  evidence: EvidenceItem[];
  recommendation: string;
  generatedAt: string;
  modelUsed: string;
  sessionId: string;
}

export interface PipelineStep {
  id: string;
  label: string;
  status: "pending" | "running" | "completed" | "error";
  duration?: number;
}

export interface AppSettings {
  confidenceThreshold: number;
  topK: number;
  modelBackend: string;
  modalityFilter: string;
}
