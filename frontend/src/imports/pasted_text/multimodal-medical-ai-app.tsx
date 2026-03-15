You are a senior frontend engineer and product designer.

Build a production-grade React application for a Multimodal Medical AI Assistant.

Important constraints:
- Framework: React
- Language: TypeScript
- Styling: Tailwind CSS
- Routing: React Router
- UI should be production-quality and responsive
- Do not use Python or Streamlit
- Assume backend APIs already exist for OCR, vision analysis, hybrid RAG retrieval, and report generation

Product context:
This platform combines:
- Medical image analysis for X-ray / CT / MRI
- OCR for lab reports and medical PDFs
- Hybrid RAG retrieval over medical knowledge
- LLM-based clinical reasoning
- Explainability with Grad-CAM heatmaps
- Structured differential diagnosis output

Design goals:
- High-trust medical UI
- Professional clinical / radiology research dashboard
- Fast workflow for clinicians and researchers
- Clear visual hierarchy
- Interactive evidence review
- Structured outputs suitable for export
- Premium medical aesthetic, not generic AI styling

Visual direction:
- Use soft clinical blues, slate, off-white, subtle steel tones
- Avoid purple-heavy AI visuals
- Clean typography, strong spacing, subtle gradients
- Panels with depth, polished cards, restrained motion
- Make it feel like a serious medical imaging and evidence platform

App structure:
Create a multi-page app with these routes/pages:

1. Dashboard / Multimodal Analysis
- Upload medical image
- Upload PDF or lab report
- Enter symptoms
- Enter clinical notes
- Select modality if needed
- Run analysis button
- Show pipeline progress states:
  - OCR
  - image analysis
  - retrieval
  - reasoning
- Show summary cards for:
  - top findings
  - confidence
  - evidence count
  - recommended next action

2. Vision Explainability
- Show uploaded image
- Show Grad-CAM heatmap overlay toggle
- Top-3 findings with confidence bars
- Image quality
- Modality
- Model backend used
- Ability to compare original image vs overlay
- Clinical-style legend for heatmap intensity

3. OCR Review
- Raw extracted text
- Structured lab values table
- Highlight abnormal values
- OCR confidence
- Search/filter within extracted content
- PDF/report metadata if available

4. Knowledge Evidence
- Retrieved literature and guideline snippets
- Evidence cards with:
  - title/source
  - metadata
  - relevance score
  - expandable snippet
- Search bar for manual evidence lookup
- Filters for source type, date, specialty, modality
- Distinguish PubMed, guidelines, coding systems, and local knowledge

5. Final Report
- Structured report layout with sections:
  - Possible Findings
  - Explanation
  - Supporting Evidence
  - Recommendation
- Show top-3 differential diagnosis clearly
- Export buttons:
  - JSON
  - text
  - print-friendly report

Core UX requirements:
- Sidebar or top control panel for:
  - confidence threshold
  - top-k retrieval
  - model/backend selection
  - modality filter
- Persist state across page navigation
- Good empty states, loading states, and error states
- Drag-and-drop uploads
- Smooth but restrained animations
- Responsive layout for desktop and tablet, usable on mobile
- Accessible colors and readable contrast
- Downloadable/exportable outputs
- Reusable components

Technical requirements:
- Use React + TypeScript + Tailwind CSS
- Use React Router for navigation
- Use modular component architecture
- Use reusable UI components for cards, tabs, tables, badges, upload zones, and progress steps
- Use local mock data if backend data is unavailable
- Add mock API service layer with typed interfaces
- Keep code clean, readable, and production-oriented
- Use realistic medical placeholder data for demo mode
- Include support for future backend integration via fetch/axios
- Prefer a clean folder structure like:
  - src/pages
  - src/components
  - src/layouts
  - src/lib
  - src/types
  - src/services
  - src/hooks

Data contracts to assume:
- Vision result:
  - topFinding
  - findings[]
  - confidence
  - modality
  - imageQuality
  - backendName
  - gradcamOverlayUrl or heatmap data
- OCR result:
  - rawText
  - confidence
  - metrics[]
- Evidence result:
  - source
  - title
  - snippet
  - relevanceScore
  - metadata
- Final report:
  - possibleFindings[]
  - explanation
  - evidence[]
  - recommendation

Output requirements:
- Generate the full React app
- Include all pages and core reusable components
- Include routing
- Include polished Tailwind styling
- Include mock data and mock service functions
- Make the app directly runnable
- Focus on frontend only, but structure it so backend integration is straightforward
