import { TrendingUp, TrendingDown, Minus, ShieldAlert, Clock, Gauge, Activity } from "lucide-react";
import type { ForecastResponse } from "@/types";

interface RiskIntelligenceStripProps {
  forecastResult: ForecastResponse | null;
  hasTraffic: boolean;
}

export function RiskIntelligenceStrip({
  forecastResult,
  hasTraffic,
}: RiskIntelligenceStripProps) {
  const currentRisk = forecastResult ? Math.round(forecastResult.current_risk * 100) : null;
  const forecastRisk = forecastResult ? Math.round(forecastResult.highest_predicted_risk * 100) : null;
  const riskDelta = (currentRisk !== null && forecastRisk !== null) ? forecastRisk - currentRisk : null;
  const horizonLabel = forecastResult?.peak_risk_horizon || (forecastResult?.horizon ? `T+${forecastResult.horizon}` : "T+5");

  // Dynamic status category
  const currentCategory = forecastResult?.current_risk_category || (currentRisk !== null ? (currentRisk >= 75 ? "CRITICAL" : currentRisk >= 50 ? "HIGH" : currentRisk >= 25 ? "MEDIUM" : "LOW") : "NO DATA");
  const overallCategory = forecastResult?.overall_risk_category || (forecastRisk !== null ? (forecastRisk >= 75 ? "CRITICAL" : forecastRisk >= 50 ? "HIGH" : forecastRisk >= 25 ? "MEDIUM" : "LOW") : "NO DATA");

  // Circular gauge calculations for current risk (radius 18, circumference ~113.1)
  const radius = 18;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = currentRisk !== null ? circumference - (currentRisk / 100) * circumference : circumference;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 my-4">
      {/* 1. CURRENT RISK PANEL */}
      <div className="intelligence-card">
        <div className="flex items-center justify-between text-muted-foreground text-[11px]">
          <span className="eyebrow">CURRENT RISK</span>
          <Gauge size={15} className="text-primary" />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <div>
            <div className="text-3xl font-extrabold font-mono tracking-tight text-foreground">
              {currentRisk !== null ? `${currentRisk}%` : "—"}
            </div>
            <div className={`mt-0.5 text-[10px] font-mono font-bold ${
              currentCategory === "CRITICAL" || currentCategory === "HIGH" ? "text-destructive" : currentCategory === "MEDIUM" ? "text-warning" : "text-emerald-500"
            }`}>
              {currentCategory}
            </div>
          </div>

          {/* Elegant Circular Radial Indicator */}
          <div className="relative h-12 w-12 flex items-center justify-center">
            <svg className="h-12 w-12 -rotate-90" viewBox="0 0 44 44">
              <circle
                cx="22"
                cy="22"
                r={radius}
                className="stroke-muted/50"
                strokeWidth="3.5"
                fill="none"
              />
              <circle
                cx="22"
                cy="22"
                r={radius}
                className={
                  currentRisk !== null && currentRisk >= 70
                    ? "stroke-destructive transition-all duration-700"
                    : currentRisk !== null && currentRisk >= 40
                    ? "stroke-warning transition-all duration-700"
                    : "stroke-emerald-400 transition-all duration-700"
                }
                strokeWidth="3.5"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="none"
              />
            </svg>
            <span className="absolute font-mono text-[9.5px] text-muted-foreground font-bold">
              {currentRisk !== null ? `${currentRisk}` : "0"}
            </span>
          </div>
        </div>

        <div className="mt-2.5 border-t border-border/50 pt-2 text-[10px] text-muted-foreground font-mono flex items-center justify-between">
          <span>Observed Baseline</span>
          <span className="text-primary font-bold">{hasTraffic ? "Active Telemetry" : "Awaiting CSV"}</span>
        </div>
      </div>

      {/* 2. FORECAST RISK PANEL (Deep Navy Card in Light Theme) */}
      <div className="intelligence-card bg-[#0B1F3A] text-white border-[#12345A] shadow-md dark:bg-card dark:text-foreground dark:border-border">
        <div className="flex items-center justify-between text-[11px]">
          <span className="font-mono text-[10px] font-bold tracking-[0.14em] uppercase text-cyan-400">
            PREDICTED RISK
          </span>
          <ShieldAlert size={15} className={forecastRisk && forecastRisk >= 70 ? "text-red-400" : "text-cyan-400"} />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <div>
            <div className={`text-3xl font-extrabold font-mono tracking-tight ${
              forecastRisk !== null && forecastRisk >= 70 ? "text-red-400" : forecastRisk !== null && forecastRisk >= 40 ? "text-amber-400" : "text-white"
            }`}>
              {forecastRisk !== null ? `${forecastRisk}%` : "—"}
            </div>
            <div className="mt-0.5 text-[10px] font-mono text-slate-300">
              Peak Stage: <span className="text-white font-bold">{overallCategory}</span>
            </div>
          </div>

          {/* Mini Trend Graph / Sparkline with Cyan trajectory */}
          <div className="h-10 w-24 flex items-end gap-1 px-1.5 py-1 rounded-md bg-[#061426] border border-[#12345A]">
            {forecastResult?.timeline && forecastResult.timeline.length > 0 ? (
              forecastResult.timeline.slice(0, 6).map((t, idx) => {
                const r = t.risk <= 1 ? t.risk * 100 : t.risk;
                const hPct = Math.max(15, Math.min(100, r));
                return (
                  <div
                    key={idx}
                    style={{ height: `${hPct}%` }}
                    className={`flex-1 rounded-t-xs transition-all ${
                      r >= 70 ? "bg-red-400" : r >= 40 ? "bg-amber-400" : "bg-cyan-400"
                    }`}
                    title={`Step ${t.step}: ${r.toFixed(0)}%`}
                  />
                );
              })
            ) : (
              <div className="w-full text-[9px] font-mono text-slate-400 text-center self-center opacity-60">
                — — —
              </div>
            )}
          </div>
        </div>

        <div className="mt-2.5 border-t border-white/15 pt-2 text-[10px] text-slate-300 font-mono flex items-center justify-between">
          <span>Lookahead Target</span>
          <span className="text-cyan-400 font-bold">{horizonLabel}</span>
        </div>
      </div>

      {/* 3. RISK TREND PANEL */}
      <div className="intelligence-card">
        <div className="flex items-center justify-between text-muted-foreground text-[11px]">
          <span className="eyebrow">RISK TREND</span>
          <Activity size={15} className="text-primary" />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <div>
            <div className={`text-3xl font-extrabold font-mono tracking-tight flex items-center gap-1.5 ${
              riskDelta !== null && riskDelta > 0
                ? "text-destructive"
                : riskDelta !== null && riskDelta < 0
                ? "text-emerald-500"
                : "text-foreground"
            }`}>
              {riskDelta !== null ? (
                <>
                  {riskDelta > 0 ? (
                    <TrendingUp size={22} className="text-destructive shrink-0" />
                  ) : riskDelta < 0 ? (
                    <TrendingDown size={22} className="text-emerald-500 shrink-0" />
                  ) : (
                    <Minus size={22} className="text-muted-foreground shrink-0" />
                  )}
                  <span>{riskDelta > 0 ? `+${riskDelta}%` : `${riskDelta}%`}</span>
                </>
              ) : (
                "—"
              )}
            </div>
            <div className="mt-0.5 text-[10px] font-mono text-muted-foreground">
              {riskDelta !== null && riskDelta > 5
                ? "Escalation Velocity"
                : riskDelta !== null && riskDelta < 0
                ? "Quenching Trajectory"
                : "Steady Envelope"}
            </div>
          </div>
        </div>

        <div className="mt-2.5 border-t border-border/50 pt-2 text-[10px] text-muted-foreground font-mono flex items-center justify-between">
          <span>Dynamics</span>
          <span className={riskDelta && riskDelta > 0 ? "text-destructive font-bold" : "text-emerald-500 font-bold"}>
            {riskDelta && riskDelta > 0 ? "Threat Escalating" : "Envelope Stable"}
          </span>
        </div>
      </div>

      {/* 4. FORECAST HORIZON PANEL */}
      <div className="intelligence-card">
        <div className="flex items-center justify-between text-muted-foreground text-[11px]">
          <span className="eyebrow">FORECAST HORIZON</span>
          <Clock size={15} className="text-primary" />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <div>
            <div className="text-3xl font-extrabold font-mono tracking-tight text-primary">
              {horizonLabel}
            </div>
            <div className="mt-0.5 text-[10px] font-mono text-muted-foreground">
              {forecastResult?.timeline ? `${forecastResult.timeline.length} Steps Synthesized` : "Autoregressive Rollout"}
            </div>
          </div>

          <div className="rounded border border-primary/30 bg-primary/10 px-2.5 py-1 font-mono text-[10.5px] text-primary font-bold">
            K=5 ROLLOUT
          </div>
        </div>

        <div className="mt-2.5 border-t border-border/50 pt-2 text-[10px] text-muted-foreground font-mono flex items-center justify-between">
          <span>LSTM Depth</span>
          <span className="text-foreground font-bold">W=20 Sequence Window</span>
        </div>
      </div>
    </div>
  );
}
