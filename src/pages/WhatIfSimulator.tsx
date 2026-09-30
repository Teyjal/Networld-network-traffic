import { useState, useRef, useMemo, useEffect } from "react";
import {
  Play,
  ShieldCheck,
  AlertCircle,
  LoaderCircle,
  Upload,
  FileUp,
  Ban,
  Lock,
  Sliders,
  Network,
  ArrowDownRight,
  ArrowUpRight,
  Info,
  RotateCcw,
  CheckCircle2,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { runCounterfactual } from "@/services/api";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import { Button } from "@/components/common/Button";
import { NetworkActivityVisualizer } from "@/components/charts/NetworkActivityVisualizer";
import { PageTitle, Panel } from "@/components/common/Panel";
import { HelpTooltip } from "@/components/common/HelpTooltip";
import type { WhatIfResponse, WhatIfStepComparison } from "@/types";

interface DefenceActionConfig {
  id: string;
  name: string;
  description: string;
  icon: typeof Ban;
  defaultTarget: string;
  targetLabel: string;
  targetPlaceholder: string;
}

const DEFENCE_ACTIONS: DefenceActionConfig[] = [
  {
    id: "Block IP",
    name: "Block IP",
    description: "Drop all inbound and outbound traffic matching the malicious IP address.",
    icon: Ban,
    defaultTarget: "185.220.101.4",
    targetLabel: "Target Malicious IP",
    targetPlaceholder: "e.g. 185.220.101.4 or 192.168.1.100",
  },
  {
    id: "Quarantine Host",
    name: "Quarantine Host",
    description: "Sever all non-essential ingress and egress connections to isolate the compromised endpoint.",
    icon: Network,
    defaultTarget: "192.168.1.105",
    targetLabel: "Target Host Endpoint",
    targetPlaceholder: "e.g. 192.168.1.105",
  },
  {
    id: "Close Port",
    name: "Close Port",
    description: "Block inbound and outbound requests on the targeted destination service port.",
    icon: Lock,
    defaultTarget: "80",
    targetLabel: "Target Service Port",
    targetPlaceholder: "e.g. 80, 445, 8080",
  },
  {
    id: "Rate Limit",
    name: "Rate Limit",
    description: "Throttle packet transmission rate and bandwidth to quench flood attack volume.",
    icon: Sliders,
    defaultTarget: "90%",
    targetLabel: "Rate Throttling Restriction",
    targetPlaceholder: "Rate limit restriction (default 90% throttle)",
  },
];

export default function WhatIfSimulator() {
  const session = useTrafficSession();
  const [selectedActionId, setSelectedActionId] = useState<string>("Block IP");
  const [targetValue, setTargetValue] = useState<string>("185.220.101.4");
  const [rateFactor, setRateFactor] = useState<number>(0.1); // 90% throttle -> 0.1 scale factor
  const [horizon, setHorizon] = useState<number>(10);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<WhatIfResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeActionConfig: DefenceActionConfig = useMemo(() => {
    return (
      DEFENCE_ACTIONS.find((a) => a.id === selectedActionId) ??
      DEFENCE_ACTIONS[0]!
    );
  }, [selectedActionId]);

  function handleActionChange(newActionId: string) {
    setSelectedActionId(newActionId);
    const cfg = DEFENCE_ACTIONS.find((a) => a.id === newActionId);
    if (cfg) {
      if (newActionId === "Rate Limit") {
        setTargetValue("90% throttle");
      } else {
        setTargetValue(cfg.defaultTarget);
      }
    }
  }

  async function simulate(file?: File) {
    if (!file && !session.activeFilename) {
      setError("Upload traffic before running a What-If defence simulation.");
      return;
    }

    setBusy(true);
    setError(null);

    let effectiveTarget: string | number = targetValue;
    if (selectedActionId === "Close Port") {
      const parsed = parseInt(targetValue, 10);
      effectiveTarget = isNaN(parsed) ? 80 : parsed;
    } else if (selectedActionId === "Rate Limit") {
      effectiveTarget = `${Math.round((1 - rateFactor) * 100)}% throttle`;
    }

    try {
      const payload: {
        action: string;
        target_value: string | number;
        horizon: number;
        rate_factor: number;
        file?: File;
        filename?: string;
      } = {
        action: selectedActionId,
        target_value: effectiveTarget,
        horizon,
        rate_factor: rateFactor,
      };
      if (file) payload.file = file;
      else if (session.activeFilename) payload.filename = session.activeFilename;

      const res = await runCounterfactual(payload);
      setResponse(res);
    } catch (err: any) {
      setError(
        err?.message || "Failed to execute What-If World Model simulation on backend."
      );
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (session.activeFilename && !response && !busy) {
      simulate();
    }
  }, [session.activeFilename]);

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true);
    setError(null);
    try {
      await trafficSession.analyzeTrafficFile(f);
      await simulate();
    } catch (err: any) {
      setError(err?.message || "Failed to process traffic CSV.");
      setBusy(false);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  // Construct chart trajectory data aligned across t+0...t+k
  const chartTrajectoryData = useMemo(() => {
    if (!response) return [];

    // Priority 1: Use step_comparisons if present
    if (response.step_comparisons && response.step_comparisons.length > 0) {
      return response.step_comparisons.map((sc) => ({
        step: sc.horizon_label,
        stepIndex: sc.step,
        baseline: Number((sc.baseline_risk * 100).toFixed(1)),
        whatif: Number((sc.whatif_risk * 100).toFixed(1)),
        delta: Number((sc.risk_delta * 100).toFixed(1)),
        baselineLevel: sc.baseline_risk_level,
        whatifLevel: sc.whatif_risk_level,
      }));
    }

    // Priority 2: Use baseline and counterfactual trajectories
    const baseTraj = response.baseline_trajectory || response.baseline?.trajectory;
    const cfTraj = response.counterfactual_trajectory || response.counterfactual?.trajectory || response.whatif_trajectory;

    if (baseTraj && cfTraj && baseTraj.length > 0) {
      return baseTraj.map((bStep, idx) => {
        const cfStep = cfTraj[idx] || bStep;
        const bRisk = bStep.risk_probability ?? (bStep.risk_percent !== undefined ? bStep.risk_percent / 100 : 0);
        const cRisk = cfStep.risk_probability ?? (cfStep.risk_percent !== undefined ? cfStep.risk_percent / 100 : 0);
        return {
          step: bStep.horizon_label || (idx === 0 ? "NOW" : `+${idx}`),
          stepIndex: bStep.step ?? idx,
          baseline: Number((bRisk * 100).toFixed(1)),
          whatif: Number((cRisk * 100).toFixed(1)),
          delta: Number(((cRisk - bRisk) * 100).toFixed(1)),
          baselineLevel: bStep.risk_level || "MEDIUM",
          whatifLevel: cfStep.risk_level || "LOW",
        };
      });
    }

    // Priority 3: Fallback from timeline items
    const baseTimeline = response.baseline?.timeline || [];
    const cfTimeline = response.counterfactual?.timeline || [];
    if (baseTimeline.length > 0) {
      return baseTimeline.map((item, idx) => {
        const cfItem = cfTimeline[idx] || item;
        const bRisk = item.risk ?? 0;
        const cRisk = cfItem.risk ?? 0;
        return {
          step: item.horizon_label || (idx === 0 ? "NOW" : `+${idx}`),
          stepIndex: item.step ?? idx,
          baseline: Number((bRisk * 100).toFixed(1)),
          whatif: Number((cRisk * 100).toFixed(1)),
          delta: Number(((cRisk - bRisk) * 100).toFixed(1)),
          baselineLevel: item.risk_category || "MEDIUM",
          whatifLevel: cfItem.risk_category || "LOW",
        };
      });
    }

    return [];
  }, [response]);

  // Extract risk metrics as numbers for visualizers
  const baselinePeak: number = response
    ? Number(((response.baseline?.peak_risk ?? response.baseline?.risk ?? 0) * 100).toFixed(1))
    : 0;
  const whatIfPeak: number = response
    ? Number(((response.counterfactual?.peak_risk ?? response.counterfactual?.risk ?? 0) * 100).toFixed(1))
    : 0;
  const riskDeltaVal = response ? response.risk_change : 0;
  const riskDeltaPct = (riskDeltaVal * 100).toFixed(1);
  const isReduced = riskDeltaVal < 0;

  return (
    <>
      <PageTitle
        title="What-If Defence Simulator"
        description="Counterfactual World Model Rollout: evaluate candidate defence actions against predicted attack trajectories."
        badge="LSTM WORLD MODEL · AUTOREGRESSIVE ROLLOUT"
      />

      {/* Model-Predicted Disclaimer Banner (Strict Requirement) */}
      <div className="mb-6 flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 p-4 text-xs text-muted-foreground shadow-sm">
        <Info size={18} className="mt-0.5 shrink-0 text-primary" />
        <div className="leading-relaxed">
          <strong className="font-semibold text-foreground">
            Model-Predicted Risk Shift Reporting Notice:
          </strong>{" "}
          This simulator reports only the neural network world model's predicted change in infiltration risk
          between baseline and counterfactually modified network states over lookaheads t+1...t+k. It does not
          claim that any defence action guarantees physical prevention of real-world cyberattacks.
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        {/* Left Column: Action Configuration */}
        <Panel
          title="Recommended Action: Test Defenses Before Enforcing"
          eyebrow="SIMULATE DEFENCE ACTION"
          action={
            <HelpTooltip
              term="What-If Simulation"
              text="Simulates the effect of applying this defensive policy without making changes to live networks."
            />
          }
        >
          {/* Action Cards Selection */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-muted-foreground">
              Select Defensive Policy:
            </span>
            <div className="grid grid-cols-2 gap-2">
              {DEFENCE_ACTIONS.map((act) => {
                const Icon = act.icon;
                const isSelected = selectedActionId === act.id;
                return (
                  <button
                    key={act.id}
                    type="button"
                    onClick={() => handleActionChange(act.id)}
                    className={`flex flex-col items-start rounded-lg border p-3 text-left transition-all ${
                      isSelected
                        ? "border-primary bg-primary/10 text-primary shadow-sm ring-1 ring-primary/40"
                        : "border-border bg-card/50 text-muted-foreground hover:border-border/80 hover:bg-muted/40 hover:text-foreground"
                    }`}
                  >
                    <div className="flex items-center gap-1.5 font-semibold text-xs text-foreground">
                      <Icon size={14} className={isSelected ? "text-primary" : "text-muted-foreground"} />
                      <span>{act.name}</span>
                    </div>
                    <span className="mt-1 line-clamp-2 text-[10px] leading-tight text-muted-foreground">
                      {act.description}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Action-Specific Parameter Inputs */}
          <div className="mt-4 space-y-3">
            {selectedActionId === "Block IP" && (
              <label className="field">
                <span className="text-xs font-medium text-foreground">Target IP Address to Block</span>
                <input
                  className="input font-mono text-xs"
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  placeholder="e.g. 185.220.101.4"
                />
                <span className="text-[10px] text-muted-foreground">
                  Zeroes out flow packet rates, flags, and packet totals associated with this IP.
                </span>
              </label>
            )}

            {selectedActionId === "Quarantine Host" && (
              <label className="field">
                <span className="text-xs font-medium text-foreground">Target Host Endpoint to Isolate</span>
                <input
                  className="input font-mono text-xs"
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  placeholder="e.g. 192.168.1.105"
                />
                <span className="text-[10px] text-muted-foreground">
                  Severs ingress and egress channels, zeroing out flags, window sizes, and active traffic.
                </span>
              </label>
            )}

            {selectedActionId === "Close Port" && (
              <label className="field">
                <span className="text-xs font-medium text-foreground">Target Port to Close</span>
                <input
                  className="input font-mono text-xs"
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  placeholder="e.g. 80, 445, 8080"
                />
                <span className="text-[10px] text-muted-foreground">
                  Blocks inbound and outbound traffic on the destination port, suppressing SYN/ACK/RST flags.
                </span>
              </label>
            )}

            {selectedActionId === "Rate Limit" && (
              <div className="space-y-2">
                <label className="field">
                  <span className="text-xs font-medium text-foreground">Rate Limit Restriction Factor</span>
                  <div className="segmented w-full">
                    {[
                      { label: "50%", factor: 0.5 },
                      { label: "75%", factor: 0.25 },
                      { label: "90%", factor: 0.1 },
                    ].map((opt) => (
                      <button
                        key={opt.label}
                        type="button"
                        className={`flex-1 text-xs ${rateFactor === opt.factor ? "active" : ""}`}
                        onClick={() => {
                          setRateFactor(opt.factor);
                          setTargetValue(`${opt.label} throttle`);
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  <span className="text-[10px] text-muted-foreground">
                    Throttles flow packet rate, byte rate, and payload sizes by {Math.round((1 - rateFactor) * 100)}%.
                  </span>
                </label>
              </div>
            )}

            {/* Lookahead Horizon K */}
            <div className="field">
              <span className="text-xs font-medium text-foreground">
                Rollout Lookahead Horizon (K steps)
              </span>
              <div className="segmented w-full">
                {[3, 5, 10, 15, 20].map((n) => (
                  <button
                    className={`flex-1 text-xs ${horizon === n ? "active" : ""}`}
                    key={n}
                    type="button"
                    onClick={() => setHorizon(n)}
                  >
                    t+{n}
                  </button>
                ))}
              </div>
              <span className="text-[10px] text-muted-foreground">
                Evaluates multi-step future state transitions S(t+1)...S(t+{horizon}) via identical LSTM World Model.
              </span>
            </div>
          </div>

          <input
            type="file"
            accept=".csv"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* Action Trigger Buttons */}
          <div className="mt-5 space-y-2">
            <Button
              className="w-full justify-center gap-2 py-2 text-xs font-semibold"
              onClick={() => simulate()}
              disabled={busy}
            >
              {busy ? (
                <LoaderCircle size={14} className="animate-spin" />
              ) : (
                <Play size={14} />
              )}
              {busy
                ? "Simulating World Model Rollout..."
                : `Simulate ${activeActionConfig.name}`}
            </Button>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex w-full items-center justify-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              <Upload size={12} />
              <span>Simulate with custom traffic CSV</span>
            </button>
          </div>

          {/* Methodological Context */}
          <div className="mt-5 rounded border border-border/80 bg-muted/20 p-3 text-[11px] leading-relaxed text-muted-foreground">
            <strong className="block font-semibold text-foreground mb-1">
              Deterministic World Model Mechanics:
            </strong>
            1. Creates an isolated copy of active 20-flow state S(t).
            <br />
            2. Perturbs physical features based on {activeActionConfig.name}.
            <br />
            3. Executes identical PyTorch LSTM autoregressive rollout on both Baseline and Modified states.
            <br />
            4. Measures predicted infiltration risk divergence across t+1...t+{horizon}.
          </div>
        </Panel>

        {/* Right Column: Results & Comparisons */}
        <div className="space-y-6">
          <Panel
            title="What Changed: Simulated Threat Trajectory"
            eyebrow="BEFORE VS AFTER DEFENCE SIMULATION"
          >
            {error ? (
              <div className="flex flex-col items-center justify-center py-16 text-center text-sm">
                <AlertCircle size={32} className="mb-2 text-warning" />
                <p className="max-w-md text-foreground">{error}</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 flex items-center gap-2 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                >
                  <Upload size={14} /> Upload Traffic CSV to Simulate
                </button>
              </div>
            ) : busy ? (
              <div className="flex h-[360px] flex-col items-center justify-center text-sm text-muted-foreground">
                <LoaderCircle size={32} className="mb-3 animate-spin text-primary" />
                <span className="font-semibold text-foreground">
                  Running Multi-Head LSTM World Model Rollout...
                </span>
                <span className="mt-1 text-xs text-muted-foreground">
                  Simulating future network conditions with and without the selected defense.
                </span>
              </div>
            ) : response && chartTrajectoryData.length > 0 ? (
              <>
                {/* Selected Action Banner */}
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-card/60 px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-primary/10 px-2 py-0.5 text-[11px] font-mono font-semibold text-primary">
                      SELECTED ACTION:
                    </span>
                    <span className="font-semibold text-foreground text-sm">
                      {response.selected_action || activeActionConfig.name}
                    </span>
                    {response.target_value && (
                      <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                        {String(response.target_value)}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="font-mono text-muted-foreground">
                      Horizon: <strong>t+1...t+{response.step_comparisons?.length ? response.step_comparisons.length - 1 : horizon}</strong>
                    </span>
                    <button
                      onClick={() => setResponse(null)}
                      className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
                    >
                      <RotateCcw size={12} /> Reset
                    </button>
                  </div>
                </div>

                {/* Key Metrics Comparison Cards */}
                <AnimatePresence>
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-5"
                  >
                    <div className="comparison rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-center">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center justify-center">
                        Threat Without Action
                        <HelpTooltip
                          term="Baseline Threat"
                          text="The peak forecasted risk if no defensive action is taken."
                        />
                      </span>
                      <b className="text-xl font-bold font-mono text-destructive">
                        {baselinePeak.toFixed(1)}%
                      </b>
                      <small className="text-[10px] text-muted-foreground">
                        {response.baseline?.risk_category || "Unmitigated"}
                      </small>
                    </div>

                    <div className="comparison rounded-lg border border-success/30 bg-success/5 p-3 text-center">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-success flex items-center justify-center">
                        Threat With Action
                        <HelpTooltip
                          term="Simulated Threat"
                          text="The peak forecasted risk after applying the selected defensive policy."
                        />
                      </span>
                      <b className="text-xl font-bold font-mono text-success">
                        {whatIfPeak.toFixed(1)}%
                      </b>
                      <small className="text-[10px] text-muted-foreground">
                        {response.counterfactual?.risk_category || "Defended"}
                      </small>
                    </div>

                    <div className="comparison comparison-featured rounded-lg border border-primary/30 bg-primary/5 p-3 text-center">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center justify-center">
                        Safety Improvement
                        <HelpTooltip
                          term="Safety Improvement"
                          text="The predicted reduction in threat level produced by this defensive action."
                        />
                      </span>
                      <b
                        className={`text-xl font-bold font-mono flex items-center justify-center gap-0.5 ${
                          isReduced ? "text-success" : "text-destructive"
                        }`}
                      >
                        {isReduced ? (
                          <ArrowDownRight size={18} />
                        ) : (
                          <ArrowUpRight size={18} />
                        )}
                        {riskDeltaPct}%
                      </b>
                      <small className="text-[10px] text-muted-foreground">
                        {isReduced ? "Predicted Risk Reduction" : "Risk Shift"}
                      </small>
                    </div>

                    <div className="comparison rounded-lg border border-border bg-card p-3 text-center">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center justify-center">
                        Protected Features
                        <HelpTooltip
                          term="Protected Features"
                          text="The count of physical flow metrics (ports, packets, flags) modified by this defense."
                        />
                      </span>
                      <b className="text-xl font-bold font-mono text-foreground">
                        {response.affected_features?.length || 18}
                      </b>
                      <small className="text-[10px] text-muted-foreground">
                        Network Flow Parameters
                      </small>
                    </div>
                  </motion.div>
                </AnimatePresence>

                {/* Trajectory Dual Chart */}
                <div className="rounded-lg border border-border/70 bg-card/40 p-4">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-border/50 pb-2">
                    <span className="text-xs font-semibold text-foreground">
                      Trajectory Comparison: Baseline vs What-If ({activeActionConfig.name})
                    </span>
                    <div className="flex items-center gap-4 text-[11px]">
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block h-2 w-4 rounded-full bg-rose-500" />
                        <span className="text-muted-foreground">Baseline (Unmitigated)</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block h-2 w-4 rounded-full bg-emerald-400" />
                        <span className="text-muted-foreground">What-If (After Defence)</span>
                      </div>
                    </div>
                  </div>

                  <div className="h-[280px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={chartTrajectoryData}
                        margin={{ top: 10, right: 16, left: -20, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient id="baseFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.25} />
                            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.0} />
                          </linearGradient>
                          <linearGradient id="whatifFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.25} />
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0.0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="step"
                          tick={{ fill: "var(--chart-text)", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fill: "var(--chart-text)", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                          unit="%"
                        />
                        <Tooltip
                          contentStyle={{
                            background: "var(--popover)",
                            border: "1px solid var(--border)",
                            borderRadius: 6,
                            fontSize: 12,
                          }}
                          formatter={(value: any, name: any) => [
                            `${value}%`,
                            name === "baseline"
                              ? "Baseline Risk"
                              : name === "whatif"
                              ? "What-If Risk"
                              : name,
                          ]}
                          labelFormatter={(label) => `Rollout Step: ${label}`}
                        />
                        {/* Operational Thresholds */}
                        <ReferenceLine
                          y={70}
                          stroke="#ef4444"
                          strokeDasharray="4 4"
                          strokeOpacity={0.6}
                          label={{ value: "HIGH (70%)", fill: "#ef4444", fontSize: 9, position: "insideTopRight" }}
                        />
                        <ReferenceLine
                          y={30}
                          stroke="#10b981"
                          strokeDasharray="4 4"
                          strokeOpacity={0.6}
                          label={{ value: "LOW (30%)", fill: "#10b981", fontSize: 9, position: "insideBottomRight" }}
                        />
                        <Area
                          type="monotone"
                          dataKey="baseline"
                          stroke="#ef4444"
                          fill="url(#baseFill)"
                          strokeWidth={2.5}
                          name="baseline"
                        />
                        <Area
                          type="monotone"
                          dataKey="whatif"
                          stroke="#10b981"
                          fill="url(#whatifFill)"
                          strokeWidth={2.5}
                          name="whatif"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* What-If Topology Simulation Visualizer */}
                <div className="mt-5">
                  <NetworkActivityVisualizer
                    currentRisk={whatIfPeak}
                    highestRisk={baselinePeak}
                    attackStage={response.counterfactual?.risk_category || "Defended"}
                    attackerIp={selectedActionId === "Block IP" ? String(targetValue) : "185.220.101.4"}
                    targetPort={selectedActionId === "Close Port" ? String(targetValue) : "80"}
                    isMitigated={isReduced}
                    mitigationAction={selectedActionId}
                  />
                </div>

                {/* Step-by-Step Risk Rollout Comparison Table (t+1...t+k) */}
                {response.step_comparisons && response.step_comparisons.length > 0 && (
                  <div className="mt-5">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-semibold text-foreground">
                        Step-by-Step Risk Predictions at t+1...t+k
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        Identical LSTM World Model forward passes
                      </span>
                    </div>

                    <div className="overflow-x-auto rounded-lg border border-border">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="border-b border-border bg-muted/40 text-[10px] text-muted-foreground uppercase">
                          <tr>
                            <th className="px-3 py-2">Lookahead</th>
                            <th className="px-3 py-2">Baseline Risk</th>
                            <th className="px-3 py-2">What-If Risk</th>
                            <th className="px-3 py-2">Predicted Shift (Δ)</th>
                            <th className="px-3 py-2">State Nature</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/60">
                          {response.step_comparisons.map((sc) => {
                            const isStepReduced = sc.risk_delta < 0;
                            return (
                              <tr
                                key={sc.step}
                                className={sc.step === 0 ? "bg-muted/10 font-semibold" : "hover:bg-muted/20"}
                              >
                                <td className="px-3 py-2 font-bold text-foreground">
                                  {sc.horizon_label}
                                  {sc.step === 0 && (
                                    <span className="ml-1.5 text-[9px] font-normal text-muted-foreground">
                                      (Current)
                                    </span>
                                  )}
                                </td>
                                <td className="px-3 py-2">
                                  <span className={sc.baseline_risk >= 0.7 ? "text-destructive font-bold" : "text-foreground"}>
                                    {sc.baseline_risk_percent}%
                                  </span>{" "}
                                  <span className="text-[10px] text-muted-foreground">
                                    ({sc.baseline_risk_level})
                                  </span>
                                </td>
                                <td className="px-3 py-2">
                                  <span className={sc.whatif_risk < 0.3 ? "text-success font-bold" : "text-foreground"}>
                                    {sc.whatif_risk_percent}%
                                  </span>{" "}
                                  <span className="text-[10px] text-muted-foreground">
                                    ({sc.whatif_risk_level})
                                  </span>
                                </td>
                                <td className="px-3 py-2">
                                  <span
                                    className={`inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px] font-bold ${
                                      isStepReduced
                                        ? "bg-success/10 text-success"
                                        : "bg-destructive/10 text-destructive"
                                    }`}
                                  >
                                    {isStepReduced ? (
                                      <ArrowDownRight size={13} />
                                    ) : (
                                      <ArrowUpRight size={13} />
                                    )}
                                    {sc.risk_delta_percent}%
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-[10px] text-muted-foreground">
                                  {sc.step === 0 ? "Observed State" : `Synthesized S(t+${sc.step})`}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Affected Network Features List */}
                {response.affected_features && response.affected_features.length > 0 && (
                  <div className="mt-5 rounded-lg border border-border bg-muted/20 p-3.5">
                    <span className="block text-xs font-semibold text-foreground mb-2">
                      Modified Network State Features ({response.affected_features.length} channels perturbed):
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {response.affected_features.map((feat) => (
                        <span
                          key={feat}
                          className="rounded border border-primary/30 bg-primary/5 px-2 py-0.5 font-mono text-[10px] text-primary"
                        >
                          {feat}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Summary Box */}
                {response.summary && (
                  <div className="mt-4 rounded border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                    <strong className="text-foreground">World Model Simulation Summary: </strong>
                    {response.summary}
                  </div>
                )}
              </>
            ) : !session.activeFilename ? (
              <div className="flex h-[360px] flex-col items-center justify-center rounded border border-dashed border-border bg-muted/5 p-6 text-center">
                <div className="mb-3 rounded-full bg-primary/10 p-3 text-primary">
                  <FileUp size={28} />
                </div>
                <span className="text-sm font-semibold tracking-wide text-foreground">
                  NO ACTIVE TRAFFIC SEQUENCE
                </span>
                <p className="mt-1 max-w-sm text-xs text-muted-foreground">
                  Upload traffic data before running counterfactual defence simulations.
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 flex items-center gap-2 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                >
                  <Upload size={14} /> Upload Traffic CSV
                </button>
              </div>
            ) : (
              <div className="flex h-[360px] flex-col items-center justify-center text-center text-sm text-muted-foreground">
                <Play size={32} className="mb-2 text-primary opacity-60" />
                <p className="font-semibold text-foreground">
                  Ready to test candidate defence action against LSTM World Model
                </p>
                <span className="mt-1 text-xs max-w-md">
                  Choose one of the 4 actions (Block IP, Quarantine Host, Close Port, Rate Limit) and click "Simulate defence" to compute multi-step future rollout comparisons.
                </span>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}
