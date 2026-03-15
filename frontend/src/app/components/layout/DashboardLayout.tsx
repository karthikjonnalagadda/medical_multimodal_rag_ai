import React, { useState } from "react";
import { Outlet, NavLink } from "react-router";
import { useAppContext } from "../../context/AppContext";
import {
  LayoutDashboard,
  Eye,
  FileText,
  BookOpen,
  ClipboardList,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  Sliders,
  MessageSquare,
} from "lucide-react";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Analysis" },
  { to: "/vision", icon: Eye, label: "Vision" },
  { to: "/ocr", icon: FileText, label: "OCR Review" },
  { to: "/evidence", icon: BookOpen, label: "Evidence" },
  { to: "/report", icon: ClipboardList, label: "Report" },
  { to: "/chat", icon: MessageSquare, label: "Chatbot" },
];

const modelOptions = [
  "Groq + Medical RAG",
  "TorchXRayVision + Groq",
  "CheXNet + Groq",
  "BioViL + Hybrid RAG",
];

export function DashboardLayout() {
  const { settings, setSettings } = useAppContext();
  const [collapsed, setCollapsed] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={`flex flex-col bg-sidebar text-sidebar-foreground transition-all duration-300 ${
          collapsed ? "w-16" : "w-64"
        }`}
      >
        {/* Logo area */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-sidebar-border">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-sidebar-primary">
            <Activity className="w-4 h-4 text-sidebar-primary-foreground" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <h1 className="text-[14px] tracking-tight text-sidebar-foreground whitespace-nowrap">
                MedAI Assistant
              </h1>
              <p className="text-[11px] text-sidebar-foreground/50">Multimodal Diagnostics</p>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 space-y-1 px-2 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-[13px] ${
                  isActive
                    ? "bg-sidebar-accent text-sidebar-primary"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                }`
              }
            >
              <item.icon className="w-[18px] h-[18px] shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Settings toggle */}
        <div className="border-t border-sidebar-border p-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-[13px] transition-colors ${
              showSettings
                ? "bg-sidebar-accent text-sidebar-primary"
                : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50"
            }`}
          >
            <Sliders className="w-[18px] h-[18px] shrink-0" />
            {!collapsed && <span>Controls</span>}
          </button>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center justify-center w-full mt-2 py-2 text-sidebar-foreground/50 hover:text-sidebar-foreground transition-colors"
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>
      </aside>

      {/* Settings panel */}
      {showSettings && (
        <aside className="w-72 bg-card border-r border-border flex flex-col overflow-y-auto">
          <div className="px-5 py-4 border-b border-border">
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-muted-foreground" />
              <h3 className="text-[14px] text-foreground">Pipeline Controls</h3>
            </div>
          </div>
          <div className="p-5 space-y-6">
            {/* Confidence */}
            <div>
              <label className="text-[12px] text-muted-foreground mb-2 block">
                Confidence Threshold
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={settings.confidenceThreshold * 100}
                onChange={(e) =>
                  setSettings({ ...settings, confidenceThreshold: Number(e.target.value) / 100 })
                }
                className="w-full accent-[#1e56a0]"
              />
              <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
                <span>0%</span>
                <span className="text-foreground">{Math.round(settings.confidenceThreshold * 100)}%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Top K */}
            <div>
              <label className="text-[12px] text-muted-foreground mb-2 block">
                Top-K Retrieval
              </label>
              <input
                type="range"
                min={1}
                max={20}
                value={settings.topK}
                onChange={(e) => setSettings({ ...settings, topK: Number(e.target.value) })}
                className="w-full accent-[#1e56a0]"
              />
              <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
                <span>1</span>
                <span className="text-foreground">{settings.topK}</span>
                <span>20</span>
              </div>
            </div>

            {/* Model Backend */}
            <div>
              <label className="text-[12px] text-muted-foreground mb-2 block">
                Model Backend
              </label>
              <select
                value={settings.modelBackend}
                onChange={(e) => setSettings({ ...settings, modelBackend: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input-background text-[13px] text-foreground"
              >
                {modelOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            {/* Modality */}
            <div>
              <label className="text-[12px] text-muted-foreground mb-2 block">
                Modality Filter
              </label>
              <select
                value={settings.modalityFilter}
                onChange={(e) => setSettings({ ...settings, modalityFilter: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input-background text-[13px] text-foreground"
              >
                <option value="all">All Modalities</option>
                <option value="xray">X-ray</option>
                <option value="ct">CT Scan</option>
                <option value="mri">MRI</option>
                <option value="ultrasound">Ultrasound</option>
              </select>
            </div>

            <div className="pt-2 border-t border-border">
              <div className="text-[11px] text-muted-foreground space-y-1">
                <p>Session: <span className="text-foreground">a7f3c2e1</span></p>
                <p>Pipeline: <span className="text-foreground">v2.4.1</span></p>
              </div>
            </div>
          </div>
        </aside>
      )}

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
