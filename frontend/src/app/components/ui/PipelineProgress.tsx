import React from "react";
import { CheckCircle2, Loader2, Circle, AlertCircle } from "lucide-react";
import type { PipelineStep } from "../../types";

interface PipelineProgressProps {
  steps: PipelineStep[];
}

export function PipelineProgress({ steps }: PipelineProgressProps) {
  return (
    <div className="flex items-center gap-1">
      {steps.map((step, i) => (
        <React.Fragment key={step.id}>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              {step.status === "completed" && (
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              )}
              {step.status === "running" && (
                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              )}
              {step.status === "pending" && (
                <Circle className="w-4 h-4 text-slate-300" />
              )}
              {step.status === "error" && (
                <AlertCircle className="w-4 h-4 text-red-500" />
              )}
              <span
                className={`text-[12px] ${
                  step.status === "completed"
                    ? "text-emerald-700"
                    : step.status === "running"
                    ? "text-blue-700"
                    : step.status === "error"
                    ? "text-red-700"
                    : "text-slate-400"
                }`}
              >
                {step.label}
              </span>
            </div>
          </div>
          {i < steps.length - 1 && (
            <div
              className={`w-8 h-px ${
                step.status === "completed" ? "bg-emerald-300" : "bg-slate-200"
              }`}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
