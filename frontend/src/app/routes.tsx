import { createBrowserRouter } from "react-router";
import { DashboardLayout } from "./components/layout/DashboardLayout";
import { MultimodalAnalysis } from "./pages/MultimodalAnalysis";
import { VisionExplainability } from "./pages/VisionExplainability";
import { OcrReview } from "./pages/OcrReview";
import { KnowledgeEvidence } from "./pages/KnowledgeEvidence";
import { FinalReport } from "./pages/FinalReport";
import { ChatbotPage } from "./pages/ChatbotPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: DashboardLayout,
    children: [
      { index: true, Component: MultimodalAnalysis },
      { path: "vision", Component: VisionExplainability },
      { path: "ocr", Component: OcrReview },
      { path: "evidence", Component: KnowledgeEvidence },
      { path: "report", Component: FinalReport },
      { path: "chat", Component: ChatbotPage },
    ],
  },
]);
