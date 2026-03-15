import React from "react";

interface ConfidenceBarProps {
  value: number;
  showLabel?: boolean;
  size?: "sm" | "md";
  className?: string;
}

export function ConfidenceBar({ value, showLabel = true, size = "md", className = "" }: ConfidenceBarProps) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80
      ? "bg-emerald-500"
      : pct >= 60
      ? "bg-blue-500"
      : pct >= 40
      ? "bg-amber-500"
      : "bg-red-500";

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className={`flex-1 rounded-full bg-slate-100 overflow-hidden ${size === "sm" ? "h-1.5" : "h-2"}`}>
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && <span className="text-[12px] text-muted-foreground w-10 text-right">{pct}%</span>}
    </div>
  );
}
