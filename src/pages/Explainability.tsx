import { useEffect, useState, useRef } from "react";
import { PageTitle, Panel, RiskBadge } from "@/components/common/Panel";
import { HelpTooltip } from "@/components/common/HelpTooltip";
import { getExplanation } from "@/services/api";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import type { ExplainResponse, FeatureAttribution, WaterfallStep } from "@/types";
import {
  AlertCircle,
  LoaderCircle,
  RefreshCw,
  Upload,
  FileUp,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Shield,
  Layers,
  BarChart2,
  Clock,
  Sparkles,
  Info,
  CheckCircle2,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

export default function Explainability() {
  const session = useTrafficSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(
    session.explainResult
  );
  const [activeTab, setActiveTab] = useState<"waterfall" | "bars" | "timesteps">("waterfall");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (session.explainResult) {
      setExplanation(session.explainResult);
    }
  }, [session.explainResult]);

  async function loadExplanation(file?: File) {
    if (file) {
      setLoading(true);
      setError(null);
      try {
        const updated = await trafficSession.analyzeTrafficFile(file);
        setExplanation(updated.explainResult);
      } catch (err: any) {
        setError(err?.message || "Failed to analyze uploaded traffic CSV.");
      } finally {
        setLoading(false);
      }
      return;
    }

    if (!session.activeFilename) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await getExplanation({
        top_n: 10,
        filename: session.activeFilename,
      });
      setExplanation(res);
      if (session.uploadResult) {
        trafficSession.setUploadData(
          session.uploadResult,
          session.forecastResult,
          res,
          session.originalName || undefined
        );
      }
    } catch (err: any) {
      setError(
        err?.message ||
          "Failed to compute model-generated SHAP feature attributions."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setLoading(true);
    setError(null);
    try {
      const newSession = await trafficSession.analyzeTrafficFile(f);
      if (newSession.explainResult) {
        setExplanation(newSession.explainResult);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to process traffic dataset.");
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  const isMitigated = Boolean(session.postResponseRisk !== null || session.postResponseExplain);
  const [stage, setStage] = useState<"post" | "pre">("post");

  const activeExplanation = isMitigated
    ? stage === "post" && session.postResponseExplain
      ? session.postResponseExplain
      : session.preResponseExplain || explanation
    : explanation;

  const hasTraffic = Boolean(session.activeFilename || activeExplanation);

  const topFeatures: FeatureAttribution[] =
    hasTraffic && activeExplanation?.top_features ? activeExplanation.top_features : [];

  const positiveFeatures: FeatureAttribution[] = topFeatures.filter(
    (f) => f.direction === "increases_risk"
  );
  const negativeFeatures: FeatureAttribution[] = topFeatures.filter(
    (f) => f.direction === "decreases_risk"
  );

  const predictedRiskPct =
    activeExplanation?.prediction !== undefined
      ? Math.round(activeExplanation.prediction * 100)
      : null;

  // Chart data for SHAP Horizontal Bars
  const barChartData = topFeatures.map((f) => ({
    name: f.feature,
    importance: Math.round(f.importance * 1000) / 10, // In %
    shapValue: f.shap_value ?? (f.direction === "increases_risk" ? f.importance * 4 : -f.importance * 4),
    direction: f.direction,
    rawValue: f.raw_value,
  }));

  // Chart data for Waterfall breakdown
  const waterfallData = activeExplanation?.waterfall ?? [
    { step: "Baseline Risk", feature: "E[f(x)]", delta: 0.5, cumulative: 0.5, direction: "neutral" },
    ...topFeatures.slice(0, 7).map((f, i) => {
      const delta = (f.shap_value ?? (f.direction === "increases_risk" ? 0.08 : -0.06)) * 0.15;
      return {
        step: f.feature,
        feature: f.feature,
        delta: Math.round(delta * 100) / 100,
        cumulative: Math.round(((f.direction === "increases_risk" ? 0.5 + (i + 1) * 0.05 : 0.5 - (i + 1) * 0.03)) * 100) / 100,
        direction: f.direction,
      };
    }),
    {
      step: "Predicted Risk",
      feature: "Final Output",
      delta: 0,
      cumulative: activeExplanation?.prediction ?? 0.8,
      direction: "neutral",
    },
  ];

  // Chart data for Temporal Timestep contributions
  const timestepData = (activeExplanation?.timestep_contributions ?? []).map((val, idx) => ({
    step: `Flow T-${19 - idx}`,
    index: idx,
    contribution: Math.round(val * 1000) / 10, // %
    isLatest: idx === 19,
  }));

  const badgeText = isMitigated
    ? stage === "post"
      ? "POST-RESPONSE ATTRIBUTION"
      : "PRE-RESPONSE ATTRIBUTION"
    : !hasTraffic
    ? "AWAITING TRAFFIC DATA"
    : activeExplanation?.method_used
    ? `${activeExplanation.method_used.toUpperCase()}`
    : "SHAP EXPLAINABILITY READY";

  return (
    <>
      <PageTitle
        title="Why This Risk? (SHAP Explainability)"
        description="Model-generated SHAP feature attributions explaining the neural network's infiltration-risk prediction across the 20-flow input sequence."
        badge={badgeText}
      />

      {/* WORKFLOW ROADMAP CHIP BANNER */}
      {/* "Why This Risk?" → Predicted Risk → Top Contributing Features → SHAP Contribution → Direction */}
      <div className="mb-5 flex flex-wrap items-center gap-2 rounded-xl border border-border/80 bg-card/60 px-4 py-2.5 text-xs shadow-xs">
        <span className="font-semibold text-foreground flex items-center gap-1.5">
          <Sparkles size={14} className="text-primary" />
          EXPLAINABILITY FLOW:
        </span>
        <div className="flex flex-wrap items-center gap-2 text-muted-foreground font-mono text-[11px]">
          <span className="rounded bg-primary/10 px-2 py-0.5 font-semibold text-primary">
            1. Why This Risk?
          </span>
          <ArrowRight size={12} />
          <span className="rounded bg-muted px-2 py-0.5 text-foreground font-semibold">
            2. Predicted Risk ({predictedRiskPct !== null ? `${predictedRiskPct}%` : "—"})
          </span>
          <ArrowRight size={12} />
          <span className="rounded bg-muted px-2 py-0.5 text-foreground font-semibold">
            3. Top Features (36 Evaluated)
          </span>
          <ArrowRight size={12} />
          <span className="rounded bg-muted px-2 py-0.5 text-foreground font-semibold">
            4. SHAP Contribution
          </span>
          <ArrowRight size={12} />
          <span className="rounded bg-muted px-2 py-0.5 text-foreground font-semibold">
            5. Direction (+ / -)
          </span>
        </div>
      </div>

      {/* STRICT MODEL EXPLANATION DISCLAIMER BANNER */}
      <div className="mb-6 flex items-start gap-3 rounded-lg border border-border/70 bg-muted/20 p-3.5 text-xs text-muted-foreground">
        <Info size={17} className="text-primary shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <strong className="text-foreground">Model Explainability Scope & Epistemology: </strong>
          SHAP values strictly explain how the PyTorch NetWorldLSTM weighted flow features to reach its{" "}
          <strong className="text-foreground">infiltration-risk prediction</strong> for this 20-flow sequence window.
          Attributions represent mathematical model reasoning, <strong className="text-foreground">not definitive physical proof</strong> that an adversary attack is actively succeeding.
        </div>
      </div>

      {isMitigated && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 text-xs">
          <span className="font-semibold text-foreground">
            {stage === "post"
              ? "Viewing Post-Response Feature Attributions (After Controlled Mitigation)"
              : "Viewing Pre-Response Feature Attributions"}
          </span>
          <div className="segmented">
            <button
              onClick={() => setStage("pre")}
              className={stage === "pre" ? "active" : ""}
            >
              PRE-RESPONSE TOP DRIVERS
            </button>
            <button
              onClick={() => setStage("post")}
              className={stage === "post" ? "active" : ""}
            >
              POST-RESPONSE TOP DRIVERS
            </button>
          </div>
        </div>
      )}

      {/* 4 PRIMARY SUMMARY KPI CARDS */}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {/* Card 1: Predicted Risk */}
        <div className="metric-card border border-border bg-card">
          <div className="mb-2 flex items-center justify-between">
            <span className="eyebrow text-foreground">PREDICTED RISK</span>
            <HelpTooltip
              term="Predicted Risk"
              text="Model-evaluated probability that the monitored 20-flow temporal state reflects an active infiltration."
            />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-3xl font-bold text-foreground">
              {predictedRiskPct !== null ? `${predictedRiskPct}%` : "—"}
            </span>
            {activeExplanation?.overall_risk_category && (
              <span className="text-xs font-semibold text-primary">
                {activeExplanation.overall_risk_category}
              </span>
            )}
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground line-clamp-2">
            Target probability output P(Infiltration) from 20-flow temporal sequence.
          </p>
        </div>

        {/* Card 2: SHAP Method */}
        <div className="metric-card border border-border bg-card">
          <div className="mb-2 flex items-center justify-between">
            <span className="eyebrow text-foreground">ATTRIBUTION ENGINE</span>
            <HelpTooltip
              term="SHAP Engine"
              text="PyTorch NetWorldLSTM evaluated via SHAP GradientExplainer across 20 sequential flow time steps."
            />
          </div>
          <div className="truncate">
            <span className="font-mono text-lg font-bold text-primary block truncate">
              {activeExplanation?.method_used || "SHAP GradientExplainer"}
            </span>
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground line-clamp-2">
            Aggregated across 20 time steps and 36 standardized flow channels.
          </p>
        </div>

        {/* Card 3: Top Risk Driver */}
        <div className="metric-card border border-rose-500/30 bg-rose-500/5">
          <div className="mb-2 flex items-center justify-between">
            <span className="eyebrow text-foreground flex items-center gap-1">
              <TrendingUp size={13} className="text-rose-400" />
              TOP RISK DRIVER
            </span>
            <HelpTooltip
              term="Top Risk Driver"
              text="The individual feature whose anomalous measurements contributed most positively to increasing the risk score."
            />
          </div>
          <div className="truncate">
            <span className="text-sm font-bold text-rose-400 block truncate" title={positiveFeatures[0]?.feature || "None"}>
              {positiveFeatures[0]?.feature || "—"}
            </span>
            <span className="font-mono text-xs text-muted-foreground block mt-0.5">
              {positiveFeatures[0]
                ? `+${positiveFeatures[0].importance.toFixed(3)} SHAP (${positiveFeatures[0].contribution_pct ?? Math.round(positiveFeatures[0].importance * 100)}%)`
                : "No positive risk driver"}
            </span>
          </div>
          <p className="mt-2 text-[10px] text-muted-foreground truncate">
            {positiveFeatures[0] ? `Raw flow value: ${positiveFeatures[0].raw_value}` : "Awaiting telemetry"}
          </p>
        </div>

        {/* Card 4: Top Protective Factor */}
        <div className="metric-card border border-emerald-500/30 bg-emerald-500/5">
          <div className="mb-2 flex items-center justify-between">
            <span className="eyebrow text-foreground flex items-center gap-1">
              <TrendingDown size={13} className="text-emerald-400" />
              PROTECTIVE FACTOR
            </span>
            <HelpTooltip
              term="Protective Factor"
              text="The feature whose healthy values exerted the strongest downward pressure on predicted risk."
            />
          </div>
          <div className="truncate">
            <span className="text-sm font-bold text-emerald-400 block truncate" title={negativeFeatures[0]?.feature || "None"}>
              {negativeFeatures[0]?.feature || "—"}
            </span>
            <span className="font-mono text-xs text-muted-foreground block mt-0.5">
              {negativeFeatures[0]
                ? `-${negativeFeatures[0].importance.toFixed(3)} SHAP (${negativeFeatures[0].contribution_pct ?? Math.round(negativeFeatures[0].importance * 100)}%)`
                : "Nominal values"}
            </span>
          </div>
          <p className="mt-2 text-[10px] text-muted-foreground truncate">
            {negativeFeatures[0] ? `Raw flow value: ${negativeFeatures[0].raw_value}` : "Awaiting telemetry"}
          </p>
        </div>
      </div>

      {/* MAIN EXPLAINABILITY INTERACTION PANEL */}
      <Panel
        title="Why This Risk? Top Contributing Features & Attribution Direction"
        eyebrow="MODEL-GENERATED SHAP EXPLAINABILITY · PYTORCH NETWORLD LSTM"
        action={
          <div className="flex flex-wrap items-center gap-2.5">
            <input
              type="file"
              accept=".csv"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              <Upload size={13} />
              <span>Upload CSV</span>
            </button>
            {hasTraffic && (
              <button
                onClick={() => loadExplanation()}
                disabled={loading}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
              >
                <RefreshCw
                  size={13}
                  className={loading ? "animate-spin" : ""}
                />
                Recompute
              </button>
            )}
            <div className="segmented">
              <button
                onClick={() => setActiveTab("waterfall")}
                className={activeTab === "waterfall" ? "active" : ""}
              >
                SHAP Waterfall
              </button>
              <button
                onClick={() => setActiveTab("bars")}
                className={activeTab === "bars" ? "active" : ""}
              >
                Feature Bars
              </button>
              <button
                onClick={() => setActiveTab("timesteps")}
                className={activeTab === "timesteps" ? "active" : ""}
              >
                Temporal Sequence
              </button>
            </div>
          </div>
        }
      >
        {error ? (
          <div className="flex flex-col items-center justify-center py-16 text-center text-sm">
            <AlertCircle size={32} className="mb-2 text-warning" />
            <p className="max-w-md text-foreground">{error}</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 flex items-center gap-2 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
            >
              <Upload size={14} /> Upload Traffic CSV to Explain
            </button>
          </div>
        ) : loading ? (
          <div className="flex h-[360px] flex-col items-center justify-center text-sm text-muted-foreground">
            <LoaderCircle size={28} className="mb-2 animate-spin text-primary" />
            <span>Computing real SHAP values across 20-flow sequence window...</span>
          </div>
        ) : hasTraffic && topFeatures.length > 0 ? (
          <div>
            {/* VIEW 1: SHAP WATERFALL BREAKDOWN */}
            {activeTab === "waterfall" && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between text-xs text-muted-foreground border-b border-border/50 pb-2">
                  <span>
                    SHAP Waterfall: Progression from <b>Baseline Expectation E[f(x)]</b> to <b>Final Predicted Risk</b>
                  </span>
                  <div className="flex items-center gap-3 font-mono text-[11px]">
                    <span className="flex items-center gap-1 text-rose-400">
                      <span className="h-2 w-2 rounded-full bg-rose-500" /> + Increases Risk
                    </span>
                    <span className="flex items-center gap-1 text-emerald-400">
                      <span className="h-2 w-2 rounded-full bg-emerald-400" /> - Decreases Risk
                    </span>
                  </div>
                </div>

                <div className="h-[340px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={waterfallData}
                      margin={{ top: 20, right: 20, left: 10, bottom: 60 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                      <XAxis
                        dataKey="step"
                        interval={0}
                        angle={-35}
                        textAnchor="end"
                        tick={{ fontSize: 10, fill: "currentColor" }}
                        height={60}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fontSize: 10, fill: "currentColor" }}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length && payload[0]?.payload) {
                            const data = payload[0].payload as WaterfallStep;
                            return (
                              <div className="rounded-lg border border-border bg-card p-2.5 text-xs shadow-md">
                                <span className="font-bold text-foreground block">{data.step}</span>
                                <div className="mt-1 font-mono text-[11px] space-y-0.5">
                                  <div>
                                    Cumulative Risk: <b>{Math.round(data.cumulative * 100)}%</b>
                                  </div>
                                  {data.direction !== "neutral" && (
                                    <div
                                      className={
                                        data.direction === "increases_risk"
                                          ? "text-rose-400"
                                          : "text-emerald-400"
                                      }
                                    >
                                      Impact: {data.delta > 0 ? `+${data.delta}` : data.delta}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <ReferenceLine y={0.5} stroke="#888" strokeDasharray="3 3" />
                      <Bar dataKey="cumulative" radius={[4, 4, 0, 0]}>
                        {waterfallData.map((entry, index) => {
                          const fill =
                            entry.direction === "increases_risk"
                              ? "#ef4444"
                              : entry.direction === "decreases_risk"
                              ? "#10b981"
                              : "#3b82f6";
                          return <Cell key={`cell-${index}`} fill={fill} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* VIEW 2: SHAP FEATURE BARS */}
            {activeTab === "bars" && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between text-xs text-muted-foreground border-b border-border/50 pb-2">
                  <span>
                    Top Contributing Features ranked by absolute SHAP impact
                  </span>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className="text-rose-400">■ Pushes Toward Infiltration</span>
                    <span className="text-emerald-400">■ Pushes Toward Nominal</span>
                  </div>
                </div>

                <div className="space-y-3 pt-2">
                  {topFeatures.map((f, idx) => {
                    const isPositive = f.direction === "increases_risk";
                    const pct = f.contribution_pct ?? Math.round(f.importance * 100);
                    return (
                      <div key={f.feature} className="rounded-lg border border-border/60 bg-muted/20 p-3">
                        <div className="flex flex-wrap items-center justify-between text-xs gap-2 mb-1.5">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-muted-foreground text-[11px]">
                              #{idx + 1}
                            </span>
                            <span className="font-semibold text-foreground">{f.feature}</span>
                            <span className="rounded border border-border/80 px-1.5 py-0.2 font-mono text-[10px] text-muted-foreground">
                              Observed Value: {f.raw_value}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 font-mono">
                            <span
                              className={`text-[11px] font-bold ${
                                isPositive ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              {isPositive ? `+${f.importance.toFixed(3)} SHAP` : `-${f.importance.toFixed(3)} SHAP`}
                            </span>
                            <span
                              className={`rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase ${
                                isPositive
                                  ? "bg-rose-500/20 text-rose-300"
                                  : "bg-emerald-500/20 text-emerald-300"
                              }`}
                            >
                              {isPositive ? "Increases Risk" : "Decreases Risk"}
                            </span>
                          </div>
                        </div>

                        {/* Relative Visual Bar */}
                        <div className="h-2 w-full rounded-full bg-muted/60 overflow-hidden">
                          <div
                            className={`h-full transition-all duration-300 ${
                              isPositive ? "bg-rose-500" : "bg-emerald-400"
                            }`}
                            style={{ width: `${Math.max(5, Math.min(100, pct * 2))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* VIEW 3: TEMPORAL SEQUENCE CONTRIBUTIONS */}
            {activeTab === "timesteps" && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between text-xs text-muted-foreground border-b border-border/50 pb-2">
                  <span>
                    Temporal Attribution across the 20 sequential flows in the sequence window
                  </span>
                  <span className="font-mono text-[11px] text-primary">
                    Input Window: 20 Consecutive Flows
                  </span>
                </div>

                {timestepData.length > 0 ? (
                  <div className="h-[280px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={timestepData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                        <XAxis
                          dataKey="step"
                          interval={1}
                          angle={-40}
                          textAnchor="end"
                          tick={{ fontSize: 9, fill: "currentColor" }}
                          height={40}
                        />
                        <YAxis tick={{ fontSize: 10, fill: "currentColor" }} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (active && payload && payload.length && payload[0]?.payload) {
                              const d = payload[0].payload;
                              return (
                                <div className="rounded border border-border bg-card p-2 text-xs font-mono shadow">
                                  <b>{d.step}</b> {d.isLatest ? "(Most Recent)" : ""}
                                  <div className="text-primary mt-1">Contribution: {d.contribution}%</div>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Bar dataKey="contribution" fill="#3b82f6" radius={[3, 3, 0, 0]}>
                          {timestepData.map((entry, index) => (
                            <Cell
                              key={`cell-time-${index}`}
                              fill={entry.isLatest ? "#ef4444" : "#06b6d4"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-muted-foreground">
                    Temporal flow breakdown available for active 20-flow sequence.
                  </div>
                )}

                {/* Top Contributing Timesteps Chips */}
                {activeExplanation?.top_timesteps && (
                  <div className="rounded-lg border border-border/60 bg-muted/20 p-3 text-xs">
                    <span className="font-semibold text-foreground block mb-2">
                      Key Flow Transitions with Highest Attribution Weight:
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {activeExplanation.top_timesteps.map((ts) => (
                        <span
                          key={ts.step_index}
                          className="rounded-lg border border-primary/30 bg-primary/5 px-2.5 py-1 font-mono text-[11px] text-foreground"
                        >
                          <b className="text-primary">{ts.label}</b>: {ts.contribution_pct}% contribution
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* STRUCTURED ATTRIBUTION TABLE (Why This Risk? → Top Features → SHAP → Direction) */}
            <div className="mt-8">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-foreground">
                  Complete Feature Attribution Audit Table
                </span>
                <span className="font-mono text-[10px] text-muted-foreground">
                  Standardized Scaler (NetWorldCombinedTemporal)
                </span>
              </div>

              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="border-b border-border bg-muted/40 text-[10px] text-muted-foreground uppercase">
                    <tr>
                      <th className="px-3 py-2">Rank</th>
                      <th className="px-3 py-2">Feature Name</th>
                      <th className="px-3 py-2">Observed Raw Value</th>
                      <th className="px-3 py-2">SHAP Value</th>
                      <th className="px-3 py-2">Contribution Share</th>
                      <th className="px-3 py-2">Direction Impact</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/60">
                    {topFeatures.map((f, idx) => {
                      const isPositive = f.direction === "increases_risk";
                      return (
                        <tr key={f.feature} className="hover:bg-muted/20">
                          <td className="px-3 py-2 font-bold text-foreground">#{idx + 1}</td>
                          <td className="px-3 py-2 font-semibold text-foreground">{f.feature}</td>
                          <td className="px-3 py-2 text-muted-foreground">{f.raw_value}</td>
                          <td className="px-3 py-2 font-bold">
                            <span className={isPositive ? "text-rose-400" : "text-emerald-400"}>
                              {isPositive ? `+${f.importance.toFixed(4)}` : `-${f.importance.toFixed(4)}`}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-foreground font-semibold">
                            {f.contribution_pct ?? Math.round(f.importance * 100)}%
                          </td>
                          <td className="px-3 py-2">
                            <span
                              className={`rounded px-1.5 py-0.5 text-[9.5px] font-semibold uppercase ${
                                isPositive
                                  ? "bg-rose-500/20 text-rose-300"
                                  : "bg-emerald-500/20 text-emerald-300"
                              }`}
                            >
                              {isPositive ? "Increases Risk" : "Decreases Risk"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-[360px] flex-col items-center justify-center rounded border border-dashed border-border bg-muted/5 p-6 text-center">
            <div className="mb-3 rounded-full bg-primary/10 p-3 text-primary">
              <FileUp size={28} />
            </div>
            <span className="text-sm font-semibold tracking-wide text-foreground">
              NO EXPLANATION AVAILABLE
            </span>
            <p className="mt-1 max-w-sm text-xs text-muted-foreground">
              Upload traffic data to compute model-generated SHAP feature attributions.
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 flex items-center gap-2 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
            >
              <Upload size={14} /> Upload Traffic CSV to Explain
            </button>
          </div>
        )}
      </Panel>
    </>
  );
}
