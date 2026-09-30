import { useState } from "react";
import { Play, RotateCcw, ShieldCheck, LoaderCircle, CheckCircle2, AlertCircle, ArrowRight, Sliders, Shield, Zap } from "lucide-react";
import { runCounterfactual } from "@/services/api";
import type { WhatIfResponse, ForecastTimelineItem } from "@/types";

interface WhatIfSplitScreenProps {
  activeFilename?: string | null;
  initialTimeline?: ForecastTimelineItem[];
  onSimulationComplete?: (res: WhatIfResponse) => void;
}

const DEFENCE_POLICIES = [
  { id: "Block IP", name: "Block Source IP", desc: "Drop all ingress flows matching malicious external source", defaultTarget: "185.220.101.4", affectedState: "Ingress edge ACL: 185.220.101.4 -> DROP" },
  { id: "Quarantine Host", name: "Quarantine Host", desc: "Sever external ingress/egress to isolate compromised DMZ endpoint", defaultTarget: "10.0.0.5", affectedState: "Host isolation: VLAN 999 sandbox applied to 10.0.0.5" },
  { id: "Close Port", name: "Close Port", desc: "Block inbound connections on targeted public ingress port", defaultTarget: "80", affectedState: "Firewall rule: PORT 80/TCP -> REJECT" },
  { id: "Rate Limit", name: "Rate Limit Flow", desc: "Throttle packet transmission rate and bandwidth by 90%", defaultTarget: "90% throttle", affectedState: "Token bucket queue: 10% egress bandwidth ceiling" },
];

export function WhatIfSplitScreen({
  activeFilename,
  initialTimeline = [],
  onSimulationComplete,
}: WhatIfSplitScreenProps) {
  const [selectedAction, setSelectedAction] = useState<string>("Close Port");
  const [targetValue, setTargetValue] = useState<string>("80");
  const [horizon, setHorizon] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WhatIfResponse | null>(null);

  const activePolicy = DEFENCE_POLICIES.find((p) => p.id === selectedAction) ?? DEFENCE_POLICIES[0]!;

  function handleActionSelect(actionId: string) {
    setSelectedAction(actionId);
    const p = DEFENCE_POLICIES.find((x) => x.id === actionId);
    if (p) setTargetValue(p.defaultTarget);
  }

  async function handleSimulate() {
    setLoading(true);
    setError(null);
    try {
      let target: string | number = targetValue;
      if (selectedAction === "Close Port") {
        const parsed = parseInt(targetValue, 10);
        target = isNaN(parsed) ? 80 : parsed;
      }

      const res = await runCounterfactual({
        action: selectedAction,
        target_value: target,
        horizon,
        rate_factor: 0.1,
        filename: activeFilename || "sample_traffic.csv",
      });

      setResult(res);
      if (onSimulationComplete) onSimulationComplete(res);
    } catch (err: any) {
      setError(err?.message || "Failed to execute What-If simulation.");
    } finally {
      setLoading(false);
    }
  }

  // Trajectory comparison steps
  const steps = result?.step_comparisons || [
    { step: 0, horizon_label: "NOW", baseline_risk: 0.7768, whatif_risk: 0.0012, risk_delta: -0.7756 },
    { step: 1, horizon_label: "T+1", baseline_risk: 0.7812, whatif_risk: 0.0015, risk_delta: -0.7797 },
    { step: 2, horizon_label: "T+2", baseline_risk: 0.7950, whatif_risk: 0.0018, risk_delta: -0.7932 },
    { step: 3, horizon_label: "T+3", baseline_risk: 0.8120, whatif_risk: 0.0022, risk_delta: -0.8098 },
    { step: 4, horizon_label: "T+4", baseline_risk: 0.8240, whatif_risk: 0.0025, risk_delta: -0.8215 },
    { step: 5, horizon_label: "T+5", baseline_risk: 0.8410, whatif_risk: 0.0030, risk_delta: -0.8380 },
  ];

  const basePeakRisk = result?.baseline?.peak_risk ?? (result?.baseline?.risk ?? 0.841);
  const whatIfPeakRisk = result?.counterfactual?.peak_risk ?? (result?.counterfactual?.risk ?? 0.003);
  const riskDelta = result?.risk_change ?? (whatIfPeakRisk - basePeakRisk);
  const riskReduced = riskDelta < 0;

  return (
    <div className="rounded-xl border border-border bg-card p-5 md:p-6 shadow-sm">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3.5 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="eyebrow text-primary">COUNTERFACTUAL DEFENCE SIMULATOR · WORLD MODEL RE-ROLLOUT</span>
          </div>
          <h3 className="text-base md:text-lg font-bold text-foreground mt-0.5 tracking-tight">
            WHAT-IF DEFENCE SIMULATOR
          </h3>
          <p className="text-xs text-muted-foreground font-mono mt-0.5">
            Test counterfactual interventions on the virtual network twin before physical deployment.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10.5px] font-mono text-muted-foreground uppercase">Selected Defence:</span>
          <span className="telemetry-tag font-bold">{activePolicy.name}</span>
        </div>
      </div>

      {/* Policy Selection Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 mb-4">
        {DEFENCE_POLICIES.map((p) => {
          const isSelected = selectedAction === p.id;
          return (
            <button
              key={p.id}
              onClick={() => handleActionSelect(p.id)}
              className={`rounded-lg border p-3 text-left transition-all cursor-pointer ${
                isSelected
                  ? "border-primary bg-primary/10 shadow-[0_0_12px_rgba(6,182,212,0.15)] ring-1 ring-primary"
                  : "border-border/60 bg-secondary/35 hover:border-primary/40 hover:bg-secondary/60"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-foreground">{p.name}</span>
                {isSelected && <span className="h-2 w-2 rounded-full bg-primary" />}
              </div>
              <p className="mt-1 text-[10.5px] text-muted-foreground line-clamp-2 leading-relaxed">
                {p.desc}
              </p>
            </button>
          );
        })}
      </div>

      {/* Target Configuration & Execute Bar */}
      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-border/70 bg-secondary/40 p-3.5 mb-5">
        <div className="flex-1 min-w-[200px]">
          <label className="text-[11px] font-mono text-muted-foreground block mb-1">
            Target Parameter (IP / Port / Throttle)
          </label>
          <input
            type="text"
            value={targetValue}
            onChange={(e) => setTargetValue(e.target.value)}
            className="input text-xs font-mono"
            placeholder="e.g. 80, 445, or 185.220.101.4"
          />
        </div>

        <div className="w-[130px]">
          <label className="text-[11px] font-mono text-muted-foreground block mb-1">
            Lookahead Horizon
          </label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(parseInt(e.target.value, 10))}
            className="input text-xs font-mono"
          >
            <option value={5}>T+5 Steps</option>
            <option value={10}>T+10 Steps</option>
          </select>
        </div>

        <button
          onClick={handleSimulate}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-xs font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-sm shadow-primary/25 disabled:opacity-50 cursor-pointer"
        >
          {loading ? (
            <LoaderCircle size={15} className="animate-spin" />
          ) : (
            <Play size={15} />
          )}
          <span>Simulate Defence</span>
        </button>

        {result && (
          <button
            onClick={() => setResult(null)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground py-2 font-mono cursor-pointer"
          >
            <RotateCcw size={13} /> Reset
          </button>
        )}
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs text-warning">
          <AlertCircle size={16} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* SPLIT-SCREEN TRAJECTORY COMPARISON (BASELINE vs WHAT-IF) */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Left: BASELINE CURRENT TRAJECTORY */}
        <div className="rounded-xl border border-destructive/35 bg-card/60 p-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-border/60 pb-2.5 mb-3">
            <span className="font-bold text-xs text-foreground flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-destructive animate-pulse" />
              BASELINE: Unmitigated Attack Trajectory
            </span>
            <span className="font-mono text-xs font-bold text-destructive">
              {(basePeakRisk * 100).toFixed(1)}% Peak
            </span>
          </div>

          <div className="space-y-2">
            {steps.slice(0, 6).map((st) => (
              <div
                key={st.step}
                className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-1.5 text-xs font-mono border border-border/30"
              >
                <div className="flex items-center gap-2">
                  <span className="font-bold text-foreground w-10">{st.horizon_label}</span>
                  <span className="text-[10px] text-muted-foreground">S(t+{st.step})</span>
                </div>
                <div className="flex items-center gap-2.5">
                  <div className="w-28 h-2 rounded bg-muted/70 overflow-hidden flex justify-end">
                    <div
                      style={{ width: `${Math.min(100, Math.max(5, st.baseline_risk * 100))}%` }}
                      className="h-full bg-destructive transition-all duration-500"
                    />
                  </div>
                  <span className="font-bold text-foreground w-12 text-right">
                    {(st.baseline_risk * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: WHAT-IF COUNTERFACTUAL TRAJECTORY */}
        <div className="rounded-xl border border-emerald-500/40 bg-emerald-500/5 p-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-emerald-500/30 pb-2.5 mb-3">
            <span className="font-bold text-xs text-emerald-500 flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
              WHAT-IF: Modified Trajectory ({activePolicy.name})
            </span>
            <span className="font-mono text-xs font-bold text-emerald-500">
              {(whatIfPeakRisk * 100).toFixed(1)}% Peak
            </span>
          </div>

          <div className="space-y-2">
            {steps.slice(0, 6).map((st) => (
              <div
                key={st.step}
                className="flex items-center justify-between rounded-lg bg-background/90 px-3 py-1.5 text-xs font-mono border border-border/40"
              >
                <div className="flex items-center gap-2">
                  <span className="font-bold text-foreground w-10">{st.horizon_label}</span>
                  <span className="text-[10px] text-muted-foreground">S'(t+{st.step})</span>
                </div>
                <div className="flex items-center gap-2.5">
                  <div className="w-28 h-2 rounded bg-muted/70 overflow-hidden flex justify-start">
                    <div
                      style={{ width: `${Math.min(100, Math.max(5, st.whatif_risk * 100))}%` }}
                      className="h-full bg-emerald-500 transition-all duration-500"
                    />
                  </div>
                  <span className="font-bold text-emerald-500 w-12 text-right">
                    {(st.whatif_risk * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Comparison Summary Strip: PREDICTED RISK CHANGE · SELECTED DEFENCE · AFFECTED NETWORK STATE */}
      <div className="mt-5 rounded-xl border border-border/80 bg-secondary/50 p-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
        <div>
          <span className="text-[10px] text-muted-foreground block uppercase font-sans font-semibold">
            PREDICTED RISK CHANGE
          </span>
          <div className={`text-base font-bold flex items-center gap-1.5 mt-0.5 ${
            riskReduced ? "text-emerald-500" : "text-destructive"
          }`}>
            <CheckCircle2 size={16} />
            {(riskDelta * 100).toFixed(1)}% Risk Shift ({riskReduced ? "Quenched" : "Elevated"})
          </div>
        </div>

        <div>
          <span className="text-[10px] text-muted-foreground block uppercase font-sans font-semibold">
            SELECTED DEFENCE
          </span>
          <div className="text-foreground font-bold mt-0.5">
            {activePolicy.name} ({targetValue})
          </div>
        </div>

        <div>
          <span className="text-[10px] text-muted-foreground block uppercase font-sans font-semibold">
            AFFECTED NETWORK STATE
          </span>
          <div className="text-primary font-semibold mt-0.5 truncate" title={activePolicy.affectedState}>
            {activePolicy.affectedState}
          </div>
        </div>
      </div>

      {/* Authoritative Model Disclaimer */}
      <div className="mt-4 flex items-start gap-2 border-t border-border/50 pt-3 text-[11px] text-muted-foreground italic font-mono leading-relaxed">
        <Shield size={14} className="text-primary shrink-0 mt-0.5" />
        <span>
          Model-Predicted Counterfactual Change: This simulation evaluates prospective risk shift computed by the
          autoregressive world model. It does not provide a physical operational guarantee that adversary exploit payloads will be prevented in live production networks.
        </span>
      </div>
    </div>
  );
}
