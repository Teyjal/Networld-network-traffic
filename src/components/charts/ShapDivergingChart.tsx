import { useState } from "react";
import { ChevronDown, ChevronUp, Sparkles, TrendingUp, TrendingDown, HelpCircle, AlertCircle } from "lucide-react";
import type { FeatureAttribution, ExplainResponse } from "@/types";

interface ShapDivergingChartProps {
  explainData: ExplainResponse | null;
  predictedRisk?: number | null;
  isLoading?: boolean;
}

export function ShapDivergingChart({
  explainData,
  predictedRisk,
  isLoading = false,
}: ShapDivergingChartProps) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  const features: FeatureAttribution[] = explainData?.top_features || [];
  const maxImportance = features.length > 0 ? Math.max(...features.map((f) => f.importance), 0.01) : 1;

  return (
    <div className="rounded-xl border border-border bg-card p-5 md:p-6 shadow-sm">
      {/* Header: WHY THIS ALERT? */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3.5 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="eyebrow text-primary">NEURAL EXPLAINABILITY · SHAP GRADIENT ATTRIBUTIONS</span>
          </div>
          <h3 className="text-base md:text-lg font-bold text-foreground mt-0.5 tracking-tight">
            WHY THIS ALERT?
          </h3>
        </div>

        {predictedRisk !== undefined && predictedRisk !== null && (
          <div className="flex items-center gap-2 rounded-md border border-primary/40 bg-primary/10 px-3 py-1 font-mono text-xs">
            <span className="text-muted-foreground">Predicted Risk:</span>
            <span className="font-bold text-primary text-sm">{Math.round(predictedRisk)}%</span>
          </div>
        )}
      </div>

      {isLoading ? (
        <div className="flex h-44 flex-col items-center justify-center text-xs text-muted-foreground">
          <Sparkles size={22} className="mb-2 animate-spin text-primary" />
          <span className="font-mono">Computing integrated gradient attributions across 20-flow temporal window...</span>
        </div>
      ) : features.length > 0 ? (
        <div>
          {/* Section: Top Risk Drivers */}
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-xs font-bold text-foreground uppercase tracking-wider">
              TOP RISK DRIVERS
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">
              Deep SHAP · 51-Feature Unified State
            </span>
          </div>

          {/* Diverging Axis Header Legend */}
          <div className="mb-3.5 flex items-center justify-between text-[10.5px] font-mono border-b border-border/50 pb-2">
            <span className="flex items-center gap-1.5 text-destructive font-semibold">
              <TrendingUp size={13} /> INCREASES RISK (Adversarial Vector)
            </span>
            <span className="hidden sm:inline-block font-mono text-muted-foreground/60 text-xs">
              ←──────── 0.00 ────────→
            </span>
            <span className="flex items-center gap-1.5 text-emerald-500 font-semibold">
              DECREASES RISK (Benign Envelope) <TrendingDown size={13} />
            </span>
          </div>

          {/* Structured Table & Diverging Visualizer */}
          <div className="space-y-2.5">
            {features.slice(0, 6).map((f) => {
              const isIncrease = f.direction === "increases_risk";
              const barWidthPct = Math.max(12, Math.min(100, (f.importance / maxImportance) * 100));

              return (
                <div
                  key={f.feature}
                  className="rounded-lg border border-border/60 bg-secondary/35 p-3 text-xs hover:border-primary/40 transition-colors"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-foreground text-xs md:text-sm font-sans">
                        {f.feature}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        Observed: <strong className="text-foreground">{typeof f.raw_value === "number" ? f.raw_value.toFixed(2) : f.raw_value}</strong>
                      </span>
                    </div>

                    <div className="flex items-center gap-3 font-mono text-xs">
                      <span className="text-muted-foreground text-[11px]">
                        Contribution: <b className="text-foreground">{(f.importance * 100).toFixed(1)}%</b>
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-[9.5px] font-bold ${
                          isIncrease
                            ? "bg-destructive/15 text-destructive border border-destructive/30"
                            : "bg-emerald-500/15 text-emerald-500 border border-emerald-500/30"
                        }`}
                      >
                        {isIncrease ? "Increases Risk" : "Decreases Risk"}
                      </span>
                    </div>
                  </div>

                  {/* Clean Diverging Bar Track */}
                  <div className="relative h-2.5 w-full rounded bg-muted/60 overflow-hidden flex">
                    {/* Left half: INCREASES RISK */}
                    <div className="w-1/2 flex justify-end pr-0.5">
                      {isIncrease && (
                        <div
                          style={{ width: `${barWidthPct}%` }}
                          className="h-full rounded-l bg-destructive shadow-sm shadow-destructive/30 transition-all duration-500"
                        />
                      )}
                    </div>

                    {/* Zero Reference Divider */}
                    <div className="w-0.5 h-full bg-border" />

                    {/* Right half: DECREASES RISK */}
                    <div className="w-1/2 flex justify-start pl-0.5">
                      {!isIncrease && (
                        <div
                          style={{ width: `${barWidthPct}%` }}
                          className="h-full rounded-r bg-emerald-500 shadow-sm shadow-emerald-500/30 transition-all duration-500"
                        />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Expandable Technical Explanation Drawer */}
          <div className="mt-4 pt-3.5 border-t border-border/60">
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="flex w-full items-center justify-between text-xs font-mono text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              <span className="flex items-center gap-1.5">
                <HelpCircle size={14} className="text-primary" />
                Technical Explanation & Temporal Attribution Methodology
              </span>
              {showTechnicalDetails ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
            </button>

            {showTechnicalDetails && (
              <div className="mt-3 rounded-lg border border-border/70 bg-secondary/60 p-3.5 text-[11px] font-mono space-y-2">
                <div className="text-muted-foreground">
                  <span className="text-primary font-bold">Attribution Method:</span> {explainData?.method_used || "Integrated Gradients / Deep SHAP"}
                </div>
                <div className="text-muted-foreground">
                  <span className="text-primary font-bold">Sequence Horizon:</span> 20-flow lookback window across 51 unified features (36 flow + 15 packet)
                </div>
                <div className="text-muted-foreground">
                  <span className="text-primary font-bold">Mathematical Definition:</span> Path-integrated gradients computed along the straight-line trajectory from neutral baseline to observed network state vector.
                </div>
                <div className="text-muted-foreground">
                  <span className="text-primary font-bold">Defensive Guidance:</span> Target features exhibiting high positive attribution (e.g., flow packet rate or port flags) using counterfactual enforcement in the What-If Simulator.
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex h-36 flex-col items-center justify-center text-center text-xs text-muted-foreground">
          <Sparkles size={22} className="mb-2 text-primary opacity-60" />
          <span className="font-semibold text-foreground">No feature attribution data loaded.</span>
          <span className="text-[11px] text-muted-foreground mt-0.5">
            Upload traffic CSV telemetry to compute Deep SHAP feature attributions for forward threat detection.
          </span>
        </div>
      )}
    </div>
  );
}
