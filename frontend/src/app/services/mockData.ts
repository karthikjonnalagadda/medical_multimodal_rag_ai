import type {
  VisionResult,
  OcrResult,
  EvidenceItem,
  FinalReportData,
  PipelineStep,
} from "../types";

export const mockVisionResult: VisionResult = {
  topFinding: "Right lower lobe consolidation",
  findings: [
    {
      id: "f1",
      name: "Right lower lobe consolidation",
      confidence: 0.92,
      severity: "high",
      description:
        "Dense opacity in the right lower lobe consistent with lobar consolidation. Air bronchograms present suggesting pneumonic process.",
    },
    {
      id: "f2",
      name: "Mild cardiomegaly",
      confidence: 0.74,
      severity: "moderate",
      description:
        "Cardiothoracic ratio approximately 0.55, suggesting mild cardiac enlargement. Consider echocardiographic correlation.",
    },
    {
      id: "f3",
      name: "Small right pleural effusion",
      confidence: 0.61,
      severity: "moderate",
      description:
        "Blunting of the right costophrenic angle consistent with a small pleural effusion. Lateral decubitus view may help confirm.",
    },
  ],
  confidence: 0.92,
  modality: "Chest X-ray (PA)",
  imageQuality: "good",
  backendName: "MedViT-L/14 + BiomedCLIP",
  gradcamOverlayUrl: null,
  originalImageUrl: "",
};

export const mockOcrResult: OcrResult = {
  rawText: `LABORATORY REPORT
Patient ID: MRN-2024-08421
Date: 2026-03-10
Ordering Physician: Dr. Sarah Chen

COMPLETE BLOOD COUNT (CBC)
WBC: 14.2 x10^3/uL [4.5-11.0] HIGH
RBC: 4.52 x10^6/uL [4.50-5.90]
Hemoglobin: 13.8 g/dL [13.5-17.5]
Hematocrit: 41.2% [38.0-50.0]
Platelets: 342 x10^3/uL [150-400]
MCV: 91.2 fL [80.0-100.0]
MCH: 30.5 pg [27.0-33.0]
MCHC: 33.5 g/dL [32.0-36.0]

BASIC METABOLIC PANEL
Sodium: 138 mEq/L [136-145]
Potassium: 4.1 mEq/L [3.5-5.1]
Chloride: 102 mEq/L [98-106]
CO2: 24 mEq/L [23-29]
BUN: 18 mg/dL [7-20]
Creatinine: 1.4 mg/dL [0.7-1.3] HIGH
Glucose: 142 mg/dL [70-100] HIGH
Calcium: 9.2 mg/dL [8.5-10.5]

INFLAMMATORY MARKERS
CRP: 8.4 mg/L [0.0-3.0] HIGH
ESR: 42 mm/hr [0-20] HIGH
Procalcitonin: 0.8 ng/mL [<0.5] HIGH`,
  confidence: 0.96,
  metrics: [
    { id: "m1", name: "WBC", value: "14.2", unit: "x10^3/uL", referenceRange: "4.5-11.0", status: "high" },
    { id: "m2", name: "RBC", value: "4.52", unit: "x10^6/uL", referenceRange: "4.50-5.90", status: "normal" },
    { id: "m3", name: "Hemoglobin", value: "13.8", unit: "g/dL", referenceRange: "13.5-17.5", status: "normal" },
    { id: "m4", name: "Hematocrit", value: "41.2", unit: "%", referenceRange: "38.0-50.0", status: "normal" },
    { id: "m5", name: "Platelets", value: "342", unit: "x10^3/uL", referenceRange: "150-400", status: "normal" },
    { id: "m6", name: "Sodium", value: "138", unit: "mEq/L", referenceRange: "136-145", status: "normal" },
    { id: "m7", name: "Potassium", value: "4.1", unit: "mEq/L", referenceRange: "3.5-5.1", status: "normal" },
    { id: "m8", name: "Creatinine", value: "1.4", unit: "mg/dL", referenceRange: "0.7-1.3", status: "high" },
    { id: "m9", name: "Glucose", value: "142", unit: "mg/dL", referenceRange: "70-100", status: "high" },
    { id: "m10", name: "CRP", value: "8.4", unit: "mg/L", referenceRange: "0.0-3.0", status: "critical" },
    { id: "m11", name: "ESR", value: "42", unit: "mm/hr", referenceRange: "0-20", status: "high" },
    { id: "m12", name: "Procalcitonin", value: "0.8", unit: "ng/mL", referenceRange: "<0.5", status: "high" },
    { id: "m13", name: "Calcium", value: "9.2", unit: "mg/dL", referenceRange: "8.5-10.5", status: "normal" },
    { id: "m14", name: "BUN", value: "18", unit: "mg/dL", referenceRange: "7-20", status: "normal" },
    { id: "m15", name: "CO2", value: "24", unit: "mEq/L", referenceRange: "23-29", status: "normal" },
    { id: "m16", name: "Chloride", value: "102", unit: "mEq/L", referenceRange: "98-106", status: "normal" },
  ],
  documentType: "Laboratory Report - CBC & BMP",
  extractedDate: "2026-03-10",
  patientId: "MRN-2024-08421",
};

export const mockEvidence: EvidenceItem[] = [
  {
    id: "e1",
    source: "PubMed",
    title: "Community-Acquired Pneumonia: Diagnostic and Treatment Updates",
    snippet:
      "Lobar consolidation on chest radiography combined with elevated CRP (>5 mg/L) and procalcitonin (>0.25 ng/mL) strongly supports bacterial pneumonia. Empirical antibiotic therapy should target common pathogens including S. pneumoniae, H. influenzae, and atypical organisms...",
    relevanceScore: 0.95,
    metadata: {
      authors: "Torres A, Cilloniz C, Niederman MS et al.",
      year: "2025",
      journal: "Lancet Respiratory Medicine",
      specialty: "Pulmonology",
      doi: "10.1016/S2213-2600(25)00112-4",
    },
  },
  {
    id: "e2",
    source: "Guidelines",
    title: "ATS/IDSA Guidelines for Community-Acquired Pneumonia in Adults",
    snippet:
      "For patients with CAP requiring hospitalization: recommend combination therapy with a beta-lactam plus a macrolide, or respiratory fluoroquinolone monotherapy. Severity assessment using CURB-65 or PSI is recommended for site-of-care decisions...",
    relevanceScore: 0.91,
    metadata: {
      authors: "Metlay JP, Waterer GW, Long AC et al.",
      year: "2024",
      journal: "American Journal of Respiratory and Critical Care Medicine",
      specialty: "Infectious Disease",
    },
  },
  {
    id: "e3",
    source: "UpToDate",
    title: "Approach to elevated inflammatory markers in suspected infection",
    snippet:
      "Procalcitonin levels >0.5 ng/mL have a sensitivity of 76% and specificity of 70% for bacterial infection. Combined with CRP elevation and leukocytosis, the positive predictive value for bacterial etiology exceeds 85%. Serial monitoring recommended for treatment response...",
    relevanceScore: 0.87,
    metadata: {
      year: "2026",
      specialty: "Internal Medicine",
    },
  },
  {
    id: "e4",
    source: "ICD-10",
    title: "J18.1 - Lobar pneumonia, unspecified organism",
    snippet:
      "Classification for lobar pneumonia when the causative organism is not identified. Includes: pneumonia, lobar NOS. Excludes: pneumonia due to identified organisms (J13-J16), pneumonia in diseases classified elsewhere (J17).",
    relevanceScore: 0.82,
    metadata: {
      specialty: "Coding",
      year: "2026",
    },
  },
  {
    id: "e5",
    source: "PubMed",
    title: "Pleural Effusion in Community-Acquired Pneumonia: Incidence and Prognostic Significance",
    snippet:
      "Parapneumonic effusions occur in 20-40% of hospitalized CAP patients. Small effusions (<10mm on lateral decubitus) typically resolve with antibiotic therapy alone. Larger effusions warrant diagnostic thoracentesis to exclude empyema...",
    relevanceScore: 0.78,
    metadata: {
      authors: "Light RW, Lee YCG",
      year: "2024",
      journal: "Chest",
      specialty: "Pulmonology",
      doi: "10.1016/j.chest.2024.01.018",
    },
  },
  {
    id: "e6",
    source: "Local KB",
    title: "Institutional Protocol: CAP Management Pathway",
    snippet:
      "Step 1: Severity assessment (CURB-65/PSI). Step 2: Blood cultures x2 before antibiotics. Step 3: Empirical therapy per local antibiogram. Step 4: De-escalation at 48-72h based on culture results. Step 5: Switch to oral therapy when hemodynamically stable and tolerating PO...",
    relevanceScore: 0.73,
    metadata: {
      specialty: "Internal Medicine",
      year: "2025",
    },
  },
];

export const mockFinalReport: FinalReportData = {
  possibleFindings: [
    {
      rank: 1,
      condition: "Community-Acquired Pneumonia (Right Lower Lobe)",
      probability: 0.89,
      supportingEvidence: [
        "Right lower lobe consolidation on CXR (confidence: 0.92)",
        "Elevated WBC: 14.2 (leukocytosis)",
        "CRP: 8.4 mg/L (elevated)",
        "Procalcitonin: 0.8 ng/mL (supports bacterial etiology)",
        "Consistent with ATS/IDSA CAP diagnostic criteria",
      ],
      icdCode: "J18.1",
    },
    {
      rank: 2,
      condition: "Parapneumonic Pleural Effusion",
      probability: 0.61,
      supportingEvidence: [
        "Small right pleural effusion on CXR (confidence: 0.61)",
        "Associated with right lower lobe consolidation",
        "Occurs in 20-40% of hospitalized CAP patients",
      ],
      icdCode: "J91.0",
    },
    {
      rank: 3,
      condition: "Mild Cardiomegaly (incidental)",
      probability: 0.52,
      supportingEvidence: [
        "Cardiothoracic ratio ~0.55 on CXR",
        "Mild degree - may be positional or technique-related",
        "Recommend echocardiographic correlation if clinically indicated",
      ],
      icdCode: "I51.7",
    },
  ],
  explanation:
    "Clinical correlation of imaging findings with laboratory results supports a primary diagnosis of community-acquired pneumonia affecting the right lower lobe. The presence of lobar consolidation with air bronchograms on chest radiography, combined with leukocytosis (WBC 14.2), significantly elevated CRP (8.4 mg/L), and procalcitonin above the bacterial infection threshold (0.8 ng/mL), provides strong evidence for a bacterial pneumonic process.\n\nThe small right pleural effusion is likely parapneumonic in nature and should be monitored. If the effusion increases or the patient fails to respond to appropriate antibiotic therapy within 48-72 hours, diagnostic thoracentesis should be considered to evaluate for complicated parapneumonic effusion or empyema.\n\nThe mild cardiomegaly is noted as an incidental finding. While the cardiothoracic ratio is borderline, this may be influenced by AP positioning or inspiration technique. Echocardiographic evaluation is recommended if there are clinical concerns for underlying cardiac disease.\n\nNotably, the creatinine is mildly elevated at 1.4 mg/dL, and the glucose is elevated at 142 mg/dL. These should be considered when selecting antibiotic therapy (dose adjustment for renal function) and in the broader clinical context (screening for diabetes if not previously diagnosed).",
  evidence: mockEvidence.slice(0, 4),
  recommendation:
    "1. Initiate empirical antibiotic therapy per ATS/IDSA guidelines for CAP (beta-lactam + macrolide or respiratory fluoroquinolone)\n2. Obtain blood cultures x2 and sputum culture before starting antibiotics\n3. Calculate CURB-65 score for severity assessment and site-of-care decision\n4. Monitor pleural effusion; consider lateral decubitus film or ultrasound if effusion increases\n5. Adjust antibiotic dosing for mildly reduced renal function (Cr 1.4)\n6. Check HbA1c given elevated glucose (142 mg/dL)\n7. Consider echocardiography for cardiomegaly evaluation if clinically indicated\n8. Reassess at 48-72 hours for clinical response and antibiotic de-escalation",
  generatedAt: "2026-03-13T14:32:00Z",
  modelUsed: "MedLLaMA-3.1-70B + BiomedCLIP + MedViT-L/14",
  sessionId: "session-a7f3c2e1",
};

export const initialPipelineSteps: PipelineStep[] = [
  { id: "ocr", label: "OCR Extraction", status: "pending" },
  { id: "vision", label: "Image Analysis", status: "pending" },
  { id: "retrieval", label: "Knowledge Retrieval", status: "pending" },
  { id: "reasoning", label: "Clinical Reasoning", status: "pending" },
];
