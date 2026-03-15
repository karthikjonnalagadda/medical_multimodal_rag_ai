import React, { useState, useRef, useEffect } from "react";
import { useAppContext } from "../context/AppContext";
import { ConfidenceBar } from "../components/ui/ConfidenceBar";
import { StatusBadge } from "../components/ui/StatusBadge";
import {
  Eye,
  ToggleLeft,
  ToggleRight,
  Monitor,
  Cpu,
  ImageIcon,
  Layers,
  Info,
} from "lucide-react";

export function VisionExplainability() {
  const { visionResult, uploadedImage } = useAppContext();
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw a mock Grad-CAM heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 512;
    canvas.height = 512;

    // Dark background simulating CXR
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, 512, 512);

    // Lighter lung fields
    const gradient1 = ctx.createRadialGradient(200, 230, 30, 200, 230, 160);
    gradient1.addColorStop(0, "rgba(60, 60, 80, 1)");
    gradient1.addColorStop(1, "rgba(30, 30, 50, 1)");
    ctx.fillStyle = gradient1;
    ctx.beginPath();
    ctx.ellipse(200, 250, 120, 170, 0, 0, Math.PI * 2);
    ctx.fill();

    const gradient2 = ctx.createRadialGradient(312, 230, 30, 312, 230, 160);
    gradient2.addColorStop(0, "rgba(60, 60, 80, 1)");
    gradient2.addColorStop(1, "rgba(30, 30, 50, 1)");
    ctx.fillStyle = gradient2;
    ctx.beginPath();
    ctx.ellipse(312, 250, 120, 170, 0, 0, Math.PI * 2);
    ctx.fill();

    // Spine/mediastinum
    ctx.fillStyle = "rgba(45, 45, 65, 1)";
    ctx.fillRect(240, 60, 32, 400);

    // RLL consolidation area
    ctx.fillStyle = "rgba(90, 90, 110, 0.7)";
    ctx.beginPath();
    ctx.ellipse(340, 340, 60, 50, 0.3, 0, Math.PI * 2);
    ctx.fill();

    if (showHeatmap) {
      // Grad-CAM overlay
      const heatGrad = ctx.createRadialGradient(340, 340, 10, 340, 340, 90);
      heatGrad.addColorStop(0, "rgba(255, 0, 0, 0.65)");
      heatGrad.addColorStop(0.3, "rgba(255, 100, 0, 0.45)");
      heatGrad.addColorStop(0.6, "rgba(255, 200, 0, 0.25)");
      heatGrad.addColorStop(1, "rgba(0, 200, 255, 0.05)");
      ctx.fillStyle = heatGrad;
      ctx.beginPath();
      ctx.ellipse(340, 340, 90, 80, 0.2, 0, Math.PI * 2);
      ctx.fill();

      // Secondary hotspot at costophrenic angle
      const heatGrad2 = ctx.createRadialGradient(370, 400, 5, 370, 400, 50);
      heatGrad2.addColorStop(0, "rgba(255, 150, 0, 0.4)");
      heatGrad2.addColorStop(1, "rgba(0, 200, 255, 0.0)");
      ctx.fillStyle = heatGrad2;
      ctx.beginPath();
      ctx.ellipse(370, 400, 50, 40, 0, 0, Math.PI * 2);
      ctx.fill();

      // Mild cardiomegaly area
      const heatGrad3 = ctx.createRadialGradient(230, 260, 5, 230, 260, 45);
      heatGrad3.addColorStop(0, "rgba(255, 200, 0, 0.25)");
      heatGrad3.addColorStop(1, "rgba(0, 200, 255, 0.0)");
      ctx.fillStyle = heatGrad3;
      ctx.beginPath();
      ctx.ellipse(230, 260, 45, 40, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [showHeatmap]);

  if (!visionResult) {
    return (
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <h1 className="text-foreground flex items-center gap-2 mb-2">
          <Eye className="w-5 h-5 text-primary" />
          Vision Explainability
        </h1>
        <div className="text-center py-20 bg-card rounded-xl border border-dashed border-border mt-6">
          <Eye className="w-12 h-12 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-[14px] text-muted-foreground">No vision results available</p>
          <p className="text-[12px] text-muted-foreground/70 mt-1">
            Run the analysis pipeline first to generate vision results
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-foreground flex items-center gap-2">
          <Eye className="w-5 h-5 text-primary" />
          Vision Explainability
        </h1>
        <p className="text-[13px] text-muted-foreground mt-1">
          Grad-CAM activation heatmap and model predictions for the uploaded medical image.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Image viewer */}
        <div className="xl:col-span-2 space-y-4">
          <div className="bg-card rounded-xl border border-border overflow-hidden">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] transition-colors ${
                    showHeatmap ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
                  }`}
                >
                  {showHeatmap ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />}
                  Grad-CAM {showHeatmap ? "ON" : "OFF"}
                </button>
                <button
                  onClick={() => setCompareMode(!compareMode)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] transition-colors ${
                    compareMode ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
                  }`}
                >
                  <Layers className="w-4 h-4" />
                  Compare
                </button>
              </div>
              <StatusBadge variant="info">{visionResult.modality}</StatusBadge>
            </div>

            {/* Canvas */}
            <div className={`${compareMode ? "grid grid-cols-2 gap-0" : ""}`}>
              <div className="relative bg-slate-900 flex items-center justify-center p-4 min-h-[420px]">
                {uploadedImage ? (
                  <>
                    <img
                      src={uploadedImage}
                      alt="Uploaded medical study"
                      className="max-w-full rounded shadow-lg"
                      style={{ maxHeight: "460px" }}
                    />
                    {showHeatmap && (
                      <div
                        className="absolute inset-4 rounded pointer-events-none"
                        style={{
                          background:
                            "radial-gradient(circle at 68% 68%, rgba(255,0,0,0.45), rgba(255,140,0,0.24) 18%, rgba(255,215,0,0.14) 30%, rgba(0,0,0,0) 46%)",
                          mixBlendMode: "screen",
                        }}
                      />
                    )}
                  </>
                ) : (
                  <canvas
                    ref={canvasRef}
                    className="max-w-full rounded shadow-lg"
                    style={{ maxHeight: "460px" }}
                  />
                )}
                {showHeatmap && (
                  <div className="absolute bottom-6 right-6 bg-black/70 rounded-lg px-3 py-2">
                    <p className="text-[10px] text-white/70 mb-1">Activation Intensity</p>
                    <div className="flex items-center gap-1">
                      <div className="w-24 h-2 rounded-full" style={{
                        background: "linear-gradient(to right, rgba(0,200,255,0.5), rgba(255,200,0,0.7), rgba(255,100,0,0.8), rgba(255,0,0,0.9))"
                      }} />
                    </div>
                    <div className="flex justify-between text-[9px] text-white/50 mt-0.5">
                      <span>Low</span><span>High</span>
                    </div>
                  </div>
                )}
              </div>
              {compareMode && (
                <div className="relative bg-slate-900 flex items-center justify-center p-4 border-l border-slate-700 min-h-[420px]">
                  {uploadedImage ? (
                    <img
                      src={uploadedImage}
                      alt="Original medical study"
                      className="max-w-full rounded shadow-lg"
                      style={{ maxHeight: "460px" }}
                    />
                  ) : (
                    <canvas
                      ref={(el) => {
                        if (!el) return;
                        const ctx = el.getContext("2d");
                        if (!ctx) return;
                        el.width = 512;
                        el.height = 512;
                        ctx.fillStyle = "#1a1a2e";
                        ctx.fillRect(0, 0, 512, 512);
                        const g1 = ctx.createRadialGradient(200, 230, 30, 200, 230, 160);
                        g1.addColorStop(0, "rgba(60,60,80,1)");
                        g1.addColorStop(1, "rgba(30,30,50,1)");
                        ctx.fillStyle = g1;
                        ctx.beginPath();
                        ctx.ellipse(200, 250, 120, 170, 0, 0, Math.PI * 2);
                        ctx.fill();
                        const g2 = ctx.createRadialGradient(312, 230, 30, 312, 230, 160);
                        g2.addColorStop(0, "rgba(60,60,80,1)");
                        g2.addColorStop(1, "rgba(30,30,50,1)");
                        ctx.fillStyle = g2;
                        ctx.beginPath();
                        ctx.ellipse(312, 250, 120, 170, 0, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.fillStyle = "rgba(45,45,65,1)";
                        ctx.fillRect(240, 60, 32, 400);
                        ctx.fillStyle = "rgba(90,90,110,0.7)";
                        ctx.beginPath();
                        ctx.ellipse(340, 340, 60, 50, 0.3, 0, Math.PI * 2);
                        ctx.fill();
                      }}
                      className="max-w-full rounded shadow-lg"
                      style={{ maxHeight: "460px" }}
                    />
                  )}
                  <div className="absolute top-6 left-6 bg-black/60 rounded px-2 py-1 text-[10px] text-white/70">
                    Original
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Meta info */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { icon: ImageIcon, label: "Quality", value: visionResult.imageQuality, variant: visionResult.imageQuality === "good" || visionResult.imageQuality === "excellent" ? "success" as const : "warning" as const },
              { icon: Monitor, label: "Modality", value: visionResult.modality, variant: "info" as const },
              { icon: Cpu, label: "Backend", value: visionResult.backendName.split(" ")[0], variant: "neutral" as const },
              { icon: Eye, label: "Overall", value: `${Math.round(visionResult.confidence * 100)}%`, variant: "success" as const },
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
        </div>

        {/* Findings panel */}
        <div className="space-y-4">
          <div className="bg-card rounded-xl border border-border p-5">
            <h3 className="text-foreground mb-4 flex items-center gap-2">
              <Info className="w-4 h-4 text-primary" />
              Top Findings
            </h3>
            <div className="space-y-4">
              {visionResult.findings.map((finding, i) => (
                <div key={finding.id} className="space-y-2">
                  <div className="flex items-start gap-2">
                    <span className="w-5 h-5 rounded-full bg-primary/10 text-primary flex items-center justify-center text-[10px] mt-0.5 shrink-0">
                      {i + 1}
                    </span>
                    <div className="flex-1">
                      <p className="text-[13px] text-foreground">{finding.name}</p>
                      <ConfidenceBar value={finding.confidence} size="sm" className="mt-1" />
                    </div>
                  </div>
                  <div className="ml-7">
                    <StatusBadge
                      variant={
                        finding.severity === "critical" ? "critical" :
                        finding.severity === "high" ? "error" :
                        finding.severity === "moderate" ? "warning" : "success"
                      }
                      dot
                    >
                      {finding.severity}
                    </StatusBadge>
                    <p className="text-[12px] text-muted-foreground mt-2 leading-relaxed">
                      {finding.description}
                    </p>
                  </div>
                  {i < visionResult.findings.length - 1 && (
                    <div className="border-b border-border pt-1" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Heatmap legend */}
          <div className="bg-card rounded-xl border border-border p-5">
            <h4 className="text-foreground mb-3 text-[13px]">Heatmap Legend</h4>
            <div className="space-y-2">
              {[
                { color: "bg-red-500", label: "High activation", desc: "Strong model attention" },
                { color: "bg-orange-400", label: "Medium activation", desc: "Moderate attention" },
                { color: "bg-yellow-400", label: "Low activation", desc: "Mild attention area" },
                { color: "bg-cyan-400", label: "Minimal", desc: "Background region" },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-sm ${item.color} shrink-0`} />
                  <div>
                    <p className="text-[12px] text-foreground">{item.label}</p>
                    <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
