import React from "react";

type BadgeVariant = "success" | "warning" | "error" | "info" | "neutral" | "critical";

const variants: Record<BadgeVariant, string> = {
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  error: "bg-red-50 text-red-700 border-red-200",
  critical: "bg-red-100 text-red-800 border-red-300",
  info: "bg-blue-50 text-blue-700 border-blue-200",
  neutral: "bg-slate-50 text-slate-600 border-slate-200",
};

interface StatusBadgeProps {
  variant: BadgeVariant;
  children: React.ReactNode;
  dot?: boolean;
}

export function StatusBadge({ variant, children, dot = false }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] border ${variants[variant]}`}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            variant === "success"
              ? "bg-emerald-500"
              : variant === "warning"
              ? "bg-amber-500"
              : variant === "error" || variant === "critical"
              ? "bg-red-500"
              : variant === "info"
              ? "bg-blue-500"
              : "bg-slate-400"
          }`}
        />
      )}
      {children}
    </span>
  );
}
