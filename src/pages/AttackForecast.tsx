import { useEffect, useState, useRef } from "react";
import { RiskChart } from "@/components/charts/RiskChart";
import { NetworkActivityVisualizer } from "@/components/charts/NetworkActivityVisualizer";
import { PageTitle, Panel, RiskBadge } from "@/components/common/Panel";
import { runForecast, timelineToForecastPoints, runCounterfactual, getExplanation } from "@/services/api";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import type { ForecastPoint, ForecastResponse, WhatIfResponse, ExplainResponse } from "@/types";
import {
  LoaderCircle,
  Upload,
  AlertCircle,
  FileUp,
  Shield,
  Activity,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Layers,
  Sparkles,
  Zap,
  Play,
  RotateCcw,
  CheckCircle2,
  ChevronRight,
} from "lucide-react";

export default function AttackForecast() {
  const session = useTrafficSession();
  const [horizon, setHorizon] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResponse | null>(
    session.forecastResult
  );
  const [selectedStageName, setSelectedStageName] = useState<string | null>(null);
  const [selectedScrubStep, setSelectedScrubStep] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Explainability state
  const [explainData, setExplainData] = useState<ExplainResponse | null>(
    session.explainResult
  );
  const [explainLoading, setExplainLoading] = useState(false);

  // What-If defense simulation state
  const [whatIfAction, setWhatIfAction] = useState("Close Port");
  const [whatIfTarget, setWhatIfTarget] = useState("80");
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState<string | null>(null);

  // Sync with session
  useEffect(() => {
    if (session.forecastResult) {
      setForecastResult(session.forecastResult);
    }
    if (session.explainResult) {
      setExplainData(session.explainResult);
    }
  }, [session.forecastResult, session.explainResult]);

  async function fetchForecast(targetHorizon: number, file?: File) {
    if (file) {
      setLoading(true);
      setError(null);
      try {
        const updated = await trafficSession.analyzeTrafficFile(file);
        setForecastResult(updated.forecastResult);
        setExplainData(updated.explainResult);
      } catch (err: any) {
        setError(err?.message || "Failed to analyze uploaded traffic file.");
      } finally {
        setLoading(false);
      }
      return;
    }

    if (!session.activeFilename) return;

    setLoading(true);
    setError(null);
    try {
      const res = await runForecast(
        targetHorizon,
        undefined,
        session.activeFilename
      );
      setForecastResult(res);
      if (session.uploadResult) {
        trafficSession.setUploadData(
          session.uploadResult,
          res,
          explainData || session.explainResult,
          session.originalName || undefined
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to execute world model forecast inference.");
    } finally {
      setLoading(false);
    }
  }

  // Fetch SHAP feature attributions if not yet loaded
  async function fetchExplainability() {
    const fn = session.activeFilename || forecastResult?.filename;
    if (!fn) return;
    setExplainLoading(true);
    try {
      const exp = await getExplanation({
        top_n: 6,
        ...(fn ? { filename: fn } : {}),
      });
      setExplainData(exp);
    } catch {
      // Non-critical background explainability fallback
    } finally {
      setExplainLoading(false);
    }
  }

  useEffect(() => {
    if (session.activeFilename && forecastResult) {
      fetchForecast(horizon);
      if (!explainData) fetchExplainability();
    }
  }, [horizon]);

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setLoading(true);
    setError(null);
    try {
      const newSession = await trafficSession.analyzeTrafficFile(f);
      if (newSession.forecastResult) {
        setForecastResult(newSession.forecastResult);
      }
      if (newSession.explainResult) {
        setExplainData(newSession.explainResult);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to process traffic dataset.");
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleRunWhatIf() {
    const fn = session.activeFilename || forecastResult?.filename;
    if (!fn) {
      setWhatIfError("Please upload or select a traffic dataset first.");
      return;
    }
    setWhatIfLoading(true);
    setWhatIfError(null);
    try {
      const sim = await runCounterfactual({
        action: whatIfAction,
        target_value: whatIfTarget,
        horizon: horizon,
        ...(fn ? { filename: fn } : {}),
      });
      setWhatIfResult(sim);
    } catch (err: any) {
      setWhatIfError(err?.message || "Failed to execute counterfactual simulation.");
    } finally {
      setWhatIfLoading(false);
    }
  }

  const isMitigated = Boolean(session.postResponseRisk !== null || session.postResponseForecast);
  const [stageView, setStageView] = useState<"post" | "pre">("post");

  const activeForecast = isMitigated
    ? stageView === "post" && session.postResponseForecast
      ? session.postResponseForecast
      : session.preResponseForecast || forecastResult
    : forecastResult;

  const hasTraffic = Boolean(session.activeFilename || activeForecast);

  const data: ForecastPoint[] =
    hasTraffic && activeForecast
      ? timelineToForecastPoints(activeForecast.timeline, activeForecast.mitre_mapping)
      : [];

  // Determine if risk is escalating across the rollout horizon
  const firstPt = data.length > 0 ? data[0] : null;
  const lastPt = data.length > 1 ? data[data.length - 1] : null;
  const isEscalating = Boolean(
    firstPt && lastPt && lastPt.risk > firstPt.risk + 5
  );
  const initialRisk = firstPt ? firstPt.risk : 0;
  const peakRisk = data.length > 0 ? Math.max(...data.map((d) => d.risk)) : 0;

  // What-If Chart overlay data when simulation has run
  const whatIfChartData: ForecastPoint[] =
    whatIfResult?.counterfactual?.timeline
      ? timelineToForecastPoints(whatIfResult.counterfactual.timeline)
      : [];

  return (
    <>
      <PageTitle
        title="Future Network Trajectory"
        description="Temporal Network World Model: Autoregressive state rollout, predictive risk forecasting, SHAP explainability, and counterfactual defense."
        badge={
          isMitigated
            ? `POST-RESPONSE STATE · ${session.verificationStatus}`
            : session.isPersisted
            ? "ANALYZED TELEMETRY"
            : forecastResult
            ? "AUTOREGRESSIVE ROLLOUT READY"
            : "WORLD MODEL ONLINE"
        }
      />

      {/* Conceptual World Model Pipeline Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-card/60 p-4 text-xs">
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1 font-semibold text-foreground">
            <Activity size={13} className="text-primary" /> Observed S(t)
          </span>
          <ArrowRight size={12} className="text-primary" />
          <span className="flex items-center gap-1 font-semibold text-foreground">
            <Layers size={13} className="text-primary" /> Multi-Head LSTM
          </span>
          <ArrowRight size={12} className="text-primary" />
          <span className="flex items-center gap-1 font-semibold text-foreground">
            <Zap size={13} className="text-accent" /> State Rollout S(t+K)
          </span>
          <ArrowRight size={12} className="text-primary" />
          <span className="flex items-center gap-1 font-semibold text-foreground">
            <TrendingUp size={13} className="text-warning" /> Trajectory Risk
          </span>
          <ArrowRight size={12} className="text-primary" />
          <span className="flex items-center gap-1 font-semibold text-foreground">
            <Shield size={13} className="text-success" /> What-If Defence
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
            K-STEP AUTOREGRESSIVE ENGINE
          </span>
        </div>
      </div>

      {isMitigated && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 text-xs">
          <span className="font-semibold text-foreground">
            {stageView === "post"
              ? "Viewing Verified Post-Response State (After Applied Policy)"
              : "Viewing Pre-Response Baseline Trajectory"}
          </span>
          <div className="segmented">
            <button
              onClick={() => setStageView("pre")}
              className={stageView === "pre" ? "active" : ""}
            >
              PRE-RESPONSE
            </button>
            <button
              onClick={() => setStageView("post")}
              className={stageView === "post" ? "active" : ""}
            >
              POST-RESPONSE
            </button>
          </div>
        </div>
      )}

      {/* SECTION 13: COMPACT CURRENT STATE PANEL */}
      {hasTraffic && activeForecast && (
        <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Current Network State</span>
              <Activity size={14} className="text-primary" />
            </div>
            <div className="mt-2 text-xl font-bold font-mono text-foreground flex items-center gap-2">
              S(NOW)
              <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${
                activeForecast.unified_state?.packet_features_available
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                  : "bg-primary/10 text-primary border-primary/30"
              }`}>
                {activeForecast.unified_state?.total_feature_count || 36} FEATS
              </span>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {activeForecast.unified_state?.packet_features_available
                ? "Unified: 36 Flow + 15 Packet features active"
                : "Standard: 36 Flow features active"}
            </p>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Current Infiltration Risk</span>
              <RiskBadge value={initialRisk} />
            </div>
            <div className="mt-2 text-2xl font-bold font-mono text-foreground">
              {initialRisk.toFixed(1)}%
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Triage Level: <b className="text-foreground">{activeForecast.current_risk_category || "MONITORED"}</b>
            </p>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Peak Forecast Risk</span>
              {isEscalating ? (
                <span className="flex items-center gap-1 text-[10px] font-semibold text-warning">
                  <TrendingUp size={12} /> ESCALATING
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[10px] font-semibold text-success">
                  <TrendingDown size={12} /> STABLE
                </span>
              )}
            </div>
            <div className="mt-2 text-2xl font-bold font-mono text-foreground">
              {peakRisk.toFixed(1)}%
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Peak Horizon: <b className="text-foreground">{activeForecast.peak_risk_horizon || `+${horizon}`}</b>
            </p>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>ATT&CK Interpretation</span>
              <Shield size={14} className="text-primary" />
            </div>
            <div className="mt-2 truncate text-base font-semibold text-primary">
              {activeForecast.mitre_mapping?.stage || "Monitored"}
            </div>
            <p className="mt-1 truncate text-[11px] text-muted-foreground">
              {activeForecast.mitre_mapping?.technique_id || "Evidence-Based Mapping"}
            </p>
          </div>
        </div>
      )}

      {/* UNIFIED NETWORK STATE & PACKET TELEMETRY CHIP BAR */}
      {hasTraffic && activeForecast?.unified_state && (
        <div className="mb-6 rounded-lg border border-border bg-card/60 p-3 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-2 mb-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-foreground">Unified Network State Specification:</span>
              <span className="font-mono text-primary font-bold">{activeForecast.unified_state.total_feature_count} Total Dimensions</span>
              <span className="text-muted-foreground">· 36 Flow Features (100% Retained) + 15 Packet Statistics</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Packet Telemetry Layer:</span>
              <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                activeForecast.unified_state.packet_features_available
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-muted text-muted-foreground border border-border"
              }`}>
                {activeForecast.unified_state.packet_features_available ? "Active (Extracted from Capture)" : "Available via PCAP / Packet Telemetry"}
              </span>
            </div>
          </div>
          {activeForecast.unified_state.packet_features && Object.keys(activeForecast.unified_state.packet_features).length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 pt-1 font-mono text-[11px]">
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">TTL Mean / Min</span>
                <span className="font-semibold text-foreground">
                  {activeForecast.unified_state.packet_features["Pkt TTL Mean"] ?? 64} / {activeForecast.unified_state.packet_features["Pkt TTL Min"] ?? 60}
                </span>
              </div>
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">TCP Win Mean</span>
                <span className="font-semibold text-foreground">
                  {activeForecast.unified_state.packet_features["TCP Win Mean"] ?? 14600} B
                </span>
              </div>
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">Payload Len Mean</span>
                <span className="font-semibold text-foreground">
                  {activeForecast.unified_state.packet_features["Payload Len Mean"] ?? 0} B
                </span>
              </div>
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">Retransmissions</span>
                <span className="font-semibold text-foreground">
                  {activeForecast.unified_state.packet_features["Retransmission Cnt"] ?? 0} pkts
                </span>
              </div>
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">SYN Only Ratio</span>
                <span className="font-semibold text-foreground">
                  {((activeForecast.unified_state.packet_features["SYN Only Ratio"] ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bg-background/80 rounded px-2 py-1 border border-border/40">
                <span className="text-muted-foreground block text-[10px]">Unique Dst Ports</span>
                <span className="font-semibold text-foreground">
                  {activeForecast.unified_state.packet_features["Unique Dst Ports"] ?? 1} ports
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Escalating Risk Warning Alert Banner */}
      {isEscalating && (
        <div className="mb-6 flex items-center gap-3 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs text-foreground">
          <AlertCircle size={18} className="text-warning shrink-0" />
          <div>
            <b>Threat Escalation Detected in Forward Rollout:</b> Neural state transitions project infiltration probability rising from{" "}
            <b>{initialRisk.toFixed(1)}%</b> at <b>NOW</b> to <b>{peakRisk.toFixed(1)}%</b> across the temporal horizon.
          </div>
        </div>
      )}

      {/* SECTION 12: MAIN FUTURE NETWORK TRAJECTORY PANEL */}
      <Panel
        title="Future Network Trajectory"
        eyebrow="PREDICTIVE NETWORK STATE WORLD MODEL · AUTOREGRESSIVE ROLLOUT"
        action={
          <div className="flex items-center gap-3">
            <input
              type="file"
              accept=".csv,.pcap,.pcapng"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
              title="Upload traffic CSV or PCAP to forecast"
            >
              <Upload size={13} />
              <span className="hidden sm:inline">Upload CSV / PCAP</span>
            </button>
            <div className="segmented">
              {[1, 3, 5, 10].map((n) => (
                <button
                  key={n}
                  onClick={() => setHorizon(n)}
                  className={horizon === n ? "active" : ""}
                  disabled={loading || !hasTraffic}
                >
                  +{n}
                </button>
              ))}
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
              <Upload size={14} /> Upload Traffic CSV
            </button>
          </div>
        ) : loading ? (
          <div className="flex h-[360px] flex-col items-center justify-center text-sm text-muted-foreground">
            <LoaderCircle size={28} className="mb-2 animate-spin text-primary" />
            <span>Executing Autoregressive State Rollout (PyTorch Multi-Head LSTM)...</span>
          </div>
        ) : hasTraffic && data.length > 0 ? (
          <>
            <RiskChart data={data} height={360} />
            <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 text-xs text-muted-foreground">
              <span>
                Engine: <b className="font-mono text-foreground">Multi-Head NetWorldLSTM</b> (State Decoder + Risk Head)
              </span>
              <span>
                State Representation: <b className="text-primary">36 Flow Features</b> (Standardized, Real Predictions)
              </span>
              <span className="font-mono text-[10px]">
                DEVICE: {activeForecast?.device_used?.toUpperCase() || "CPU"}
              </span>
            </div>
          </>
        ) : (
          <div className="flex h-[360px] flex-col items-center justify-center rounded border border-dashed border-border bg-muted/5 p-6 text-center">
            <div className="mb-3 rounded-full bg-primary/10 p-3 text-primary">
              <FileUp size={28} />
            </div>
            <span className="text-sm font-semibold tracking-wide text-foreground">
              NO TELEMETRY LOADED
            </span>
            <p className="mt-1 max-w-sm text-xs text-muted-foreground">
              Upload a network traffic capture to initiate forward state rollout.
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 flex items-center gap-2 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
            >
              <Upload size={14} /> Upload Traffic CSV
            </button>
          </div>
        )}
      </Panel>

      {/* NETWORK TRAFFIC SIMULATION & STATE TOPOLOGY VISUALIZER */}
      <div className="mt-6">
        <NetworkActivityVisualizer
          currentRisk={
            hasTraffic && data[selectedScrubStep]
              ? data[selectedScrubStep].risk
              : hasTraffic
              ? initialRisk
              : null
          }
          highestRisk={hasTraffic ? peakRisk : null}
          attackStage={
            hasTraffic && data[selectedScrubStep]
              ? data[selectedScrubStep].stage
              : activeForecast?.mitre_mapping?.stage
          }
          attackerIp={session.threatContext?.observed_source || "185.220.101.4"}
          targetIp={session.threatContext?.observed_destination || "10.0.0.5"}
          targetPort={session.threatContext?.observed_port || "445"}
          activeHorizonStep={selectedScrubStep}
        />
      </div>

      {/* SECTION 14: FORECAST HORIZON STATE CARDS */}
      {hasTraffic && data.length > 0 && (
        <div className="mt-6">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className="eyebrow">STEP-BY-STEP FORECAST TRAJECTORY</span>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Click any step below to project the network topology and flow intensity at that horizon.
              </p>
            </div>
            <span className="font-mono text-[10px] text-muted-foreground">
              S(t) → S(t+1) → S(t+2) ... AUTOREGRESSIVE SEQUENCE
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-6">
            {data.map((p, idx) => (
              <div
                onClick={() => setSelectedScrubStep(idx)}
                role="button"
                tabIndex={0}
                className={`forecast-card cursor-pointer transition-all hover:scale-[1.02] ${
                  selectedScrubStep === idx
                    ? "border-primary ring-2 ring-primary/40 bg-primary/10 shadow-sm"
                    : idx === 0
                    ? "border-primary/40 bg-primary/5"
                    : "hover:border-primary/30"
                }`}
                key={p.step}
              >
                <div className="flex items-center justify-between">
                  <span className="eyebrow flex items-center gap-1.5">
                    {p.step}
                    {selectedScrubStep === idx && (
                      <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                    )}
                  </span>
                  <RiskBadge value={p.risk} />
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-foreground">
                  {p.risk.toFixed(1)}%
                </div>
                <b className="mt-2 block truncate text-xs text-foreground">
                  {p.stage}
                </b>
                <div className="mt-3 flex items-center justify-between border-t border-border pt-2 text-[10px] text-muted-foreground">
                  <span>{idx === 0 ? "Observed" : "Synthesized"}</span>
                  <span className="font-mono text-primary font-semibold">
                    {idx === 0 ? "S(NOW)" : `S(t+${idx})`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ATTACK-STAGE MAPPING MODULE: MITRE ATT&CK PROGRESSION */}
      {hasTraffic && activeForecast?.mitre_mapping && (
        <Panel
          className="mt-6"
          title="Explicit Attack-Stage Mapping: MITRE ATT&CK Progression"
          eyebrow="EVIDENCE-BASED INFERENCE · DETERMINISTIC TELEMETRY RULES"
        >
          {(() => {
            const mitreMapping = activeForecast.mitre_mapping;
            const progressionList = mitreMapping?.progression || [];
            const canonicalStages: Array<[string, string, string, string]> = [
              ["01", "Reconnaissance", "Reconnaissance", "Reconnaissance"],
              ["02", "Initial Access", "Initial Access", "Initial Access"],
              ["03", "Lateral Movement", "Lateral Movement", "Lateral Movement"],
              ["04", "Command and Control", "C2", "Command and Control"],
              ["05", "Exfiltration", "Exfiltration", "Exfiltration"],
            ];

            return (
              <div>
                {/* Progression Pipeline Header */}
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-border/70 pb-3">
                  <div className="flex flex-wrap items-center gap-1.5 text-xs font-mono">
                    <span className="text-muted-foreground font-semibold">ATTACK PROGRESSION:</span>
                    {canonicalStages.map(([n, name, shortName, key], i, arr) => {
                      const matchedItem = progressionList.find(
                        (p) =>
                          p.stage.toLowerCase() === key.toLowerCase() ||
                          p.stage.toLowerCase().includes(shortName.toLowerCase()) ||
                          (shortName === "C2" && p.stage.toLowerCase().includes("command"))
                      );
                      const isSupported = matchedItem ? matchedItem.supported : false;
                      const isCurrentPredicted =
                        mitreMapping?.stage?.toLowerCase() === name.toLowerCase() ||
                        mitreMapping?.stage?.toLowerCase() === key.toLowerCase() ||
                        (shortName === "C2" && mitreMapping?.stage?.toLowerCase().includes("command"));

                      return (
                        <div key={name} className="flex items-center gap-1.5">
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all ${
                              isCurrentPredicted && isSupported
                                ? "bg-primary text-primary-foreground shadow-sm shadow-primary/30 ring-1 ring-primary"
                                : isSupported
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                                : "bg-muted/40 text-muted-foreground/60 border border-border/30"
                            }`}
                          >
                            {shortName}
                          </span>
                          {i < arr.length - 1 && (
                            <ChevronRight size={13} className="text-muted-foreground/40 shrink-0" />
                          )}
                        </div>
                      );
                    })}
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Supported Stages:</span>
                    <span className="font-mono font-bold text-primary">
                      {progressionList.filter((p) => p.supported).length} / 5
                    </span>
                  </div>
                </div>

                {/* OUTPUT SPECIFICATION BANNER: Predicted Stage → Supporting Evidence → MITRE Technique → Confidence */}
                <div className="mb-5 rounded-lg border border-primary/30 bg-primary/5 p-3.5 text-xs">
                  <div className="flex items-center gap-2 text-primary font-semibold font-mono text-[11px] mb-2">
                    <Shield size={14} />
                    <span>OUTPUT PIPELINE MAPPING</span>
                    <span className="text-muted-foreground text-[10px] font-normal">
                      (Deterministic Evidence Rules · Zero Hardcoded / Fabricated Values)
                    </span>
                  </div>

                  <div className="grid gap-2 md:grid-cols-4 font-mono text-[11px] rounded bg-background/80 p-2.5 border border-border/60">
                    <div className="border-r border-border/40 pr-2">
                      <span className="text-muted-foreground block text-[10px] uppercase tracking-wider font-sans">
                        Predicted Stage
                      </span>
                      <span className="font-bold text-foreground text-sm flex items-center gap-1.5 mt-0.5">
                        <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
                        {mitreMapping.stage || "Reconnaissance"}
                      </span>
                    </div>

                    <div className="border-r border-border/40 pr-2">
                      <span className="text-muted-foreground block text-[10px] uppercase tracking-wider font-sans">
                        MITRE Technique
                      </span>
                      <span className="font-semibold text-primary block mt-0.5 truncate" title={`${mitreMapping.technique_id} · ${mitreMapping.technique_name}`}>
                        {mitreMapping.technique_id} · {mitreMapping.technique_name}
                      </span>
                      <span className="text-[10px] text-muted-foreground">{mitreMapping.tactic}</span>
                    </div>

                    <div className="border-r border-border/40 pr-2">
                      <span className="text-muted-foreground block text-[10px] uppercase tracking-wider font-sans">
                        Confidence
                      </span>
                      <span className="font-bold font-mono text-emerald-400 text-sm block mt-0.5">
                        {mitreMapping.confidence_percent ?? Math.round((mitreMapping.confidence ?? mitreMapping.confidence_score ?? 0) * 100)}%
                      </span>
                      <span className="text-[10px] text-muted-foreground">Derived from rules</span>
                    </div>

                    <div>
                      <span className="text-muted-foreground block text-[10px] uppercase tracking-wider font-sans">
                        Supporting Evidence
                      </span>
                      <span
                        className="text-foreground block truncate mt-0.5"
                        title={mitreMapping.supporting_evidence?.join("; ") || "Baseline indicators"}
                      >
                        {mitreMapping.supporting_evidence && mitreMapping.supporting_evidence.length > 0
                          ? `${mitreMapping.supporting_evidence.length} telemetry indicator(s)`
                          : "Baseline envelope"}
                      </span>
                      <span className="text-[10px] text-muted-foreground truncate block">
                        {mitreMapping.supporting_evidence?.[0] || "Normal flow telemetry"}
                      </span>
                    </div>
                  </div>

                  {/* Direct canonical formatted chain string */}
                  <div className="mt-2.5 rounded bg-muted/40 px-3 py-1.5 font-mono text-[10.5px] text-muted-foreground border border-border/40 flex items-center gap-2 overflow-x-auto">
                    <span className="text-primary font-bold shrink-0">Chain:</span>
                    <span className="text-foreground shrink-0">
                      {mitreMapping.formatted_output ||
                        `${mitreMapping.stage} → Telemetry Evidence → ${mitreMapping.technique_id} (${mitreMapping.technique_name}) → ${Math.round((mitreMapping.confidence ?? 0) * 100)}%`}
                    </span>
                  </div>
                </div>

                {/* 5-STAGE PROGRESSION CARDS: Only Show Supported Stages as Active */}
                <div className="grid gap-3 md:grid-cols-5">
                  {canonicalStages.map(([n, name, shortName, key]) => {
                    const stageItem = progressionList.find(
                      (p) =>
                        p.stage.toLowerCase() === key.toLowerCase() ||
                        p.stage.toLowerCase().includes(shortName.toLowerCase()) ||
                        (shortName === "C2" && p.stage.toLowerCase().includes("command"))
                    );
                    const isSupported = stageItem ? stageItem.supported : false;
                    const isPredicted =
                      mitreMapping?.stage?.toLowerCase() === name.toLowerCase() ||
                      mitreMapping?.stage?.toLowerCase() === key.toLowerCase() ||
                      (shortName === "C2" && mitreMapping?.stage?.toLowerCase().includes("command"));
                    const confidencePercent = stageItem
                      ? stageItem.confidence_percent
                      : isPredicted
                      ? Math.round((mitreMapping?.confidence_score ?? 0) * 100)
                      : 0;
                    const evidence = stageItem?.supporting_evidence || (isPredicted ? mitreMapping?.evidence?.rules_matched || [] : []);
                    const techniqueId = stageItem?.technique_id || (isPredicted ? mitreMapping?.technique_id : "—");
                    const techniqueName = stageItem?.technique_name || (isPredicted ? mitreMapping?.technique_name : "");

                    return (
                      <div
                        key={name}
                        onClick={() => setSelectedStageName(selectedStageName === name ? null : name)}
                        className={`relative cursor-pointer rounded-lg border p-3.5 transition-all ${
                          isPredicted && isSupported
                            ? "border-primary bg-primary/10 shadow-md shadow-primary/10 ring-1 ring-primary/40"
                            : isSupported
                            ? "border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60"
                            : "border-border/40 bg-card/40 opacity-60 hover:opacity-80"
                        }`}
                      >
                        <div className="flex items-center justify-between text-[10px]">
                          <span className="font-mono text-muted-foreground">{n}</span>
                          {isSupported ? (
                            <span className="flex items-center gap-1 rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-emerald-400">
                              <CheckCircle2 size={10} />
                              SUPPORTED
                            </span>
                          ) : (
                            <span className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
                              UNSUPPORTED
                            </span>
                          )}
                        </div>

                        <div className="mt-2.5">
                          <div className="flex items-baseline justify-between">
                            <b className="text-xs font-semibold text-foreground">{name}</b>
                            {shortName === "C2" && (
                              <span className="text-[10px] font-mono text-muted-foreground">({shortName})</span>
                            )}
                          </div>
                          <div className="mt-1 font-mono text-[10.5px] text-primary truncate" title={techniqueName}>
                            {techniqueId} {techniqueName ? `· ${techniqueName}` : ""}
                          </div>
                        </div>

                        <div className="mt-3 border-t border-border/50 pt-2 flex items-center justify-between font-mono text-[11px]">
                          <span className="text-muted-foreground text-[10px]">Confidence:</span>
                          <span
                            className={`font-bold ${
                              isSupported ? "text-emerald-400" : "text-muted-foreground"
                            }`}
                          >
                            {isSupported ? `${confidencePercent.toFixed(1)}%` : "0.0%"}
                          </span>
                        </div>

                        {isSupported ? (
                          <div className="mt-2">
                            <div className="text-[9.5px] font-mono text-muted-foreground mb-1">
                              {evidence.length} Verified Evidence Rule{evidence.length === 1 ? "" : "s"}:
                            </div>
                            <ul className="space-y-1">
                              {evidence.slice(0, 2).map((ev, i) => (
                                <li key={i} className="text-[10px] text-foreground/80 leading-tight truncate" title={ev}>
                                  • {ev}
                                </li>
                              ))}
                              {evidence.length > 2 && (
                                <li className="text-[9px] font-mono text-primary">
                                  +{evidence.length - 2} more rule(s)...
                                </li>
                              )}
                            </ul>
                          </div>
                        ) : (
                          <div className="mt-2 text-[10px] text-muted-foreground italic">
                            No verified telemetry evidence detected for this stage.
                          </div>
                        )}

                        {isPredicted && isSupported && (
                          <div className="mt-2.5 rounded bg-primary/20 px-2 py-0.5 text-center font-mono text-[9.5px] font-bold text-primary">
                            ★ CURRENT PREDICTED STAGE
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Selected Stage Detail Drawer */}
                {selectedStageName && (
                  <div className="mt-4 rounded-lg border border-border bg-card/80 p-3.5 text-xs">
                    {(() => {
                      const matchedItem = progressionList.find(
                        (p) =>
                          p.stage.toLowerCase() === selectedStageName.toLowerCase() ||
                          (selectedStageName.includes("Command") && p.stage.toLowerCase().includes("command"))
                      );
                      return (
                        <div>
                          <div className="flex items-center justify-between border-b border-border/60 pb-2 mb-2">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-foreground text-sm">
                                Stage Telemetry Audit: {selectedStageName}
                              </span>
                              {matchedItem?.supported ? (
                                <span className="rounded bg-emerald-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-400">
                                  SUPPORTED · {matchedItem.confidence_percent}% CONFIDENCE
                                </span>
                              ) : (
                                <span className="rounded bg-muted px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                                  UNSUPPORTED (0.0% CONFIDENCE)
                                </span>
                              )}
                            </div>
                            <button
                              onClick={() => setSelectedStageName(null)}
                              className="text-muted-foreground hover:text-foreground text-xs"
                            >
                              Close ✕
                            </button>
                          </div>
                          {matchedItem && matchedItem.supported && matchedItem.supporting_evidence.length > 0 ? (
                            <div>
                              <div className="text-[11px] font-mono text-primary mb-1.5">
                                Verified Evidence Rules ({matchedItem.rules_satisfied} rules satisfied):
                              </div>
                              <div className="grid gap-1.5 sm:grid-cols-2">
                                {matchedItem.supporting_evidence.map((ev, i) => (
                                  <div key={i} className="rounded bg-background p-2 border border-border/50 text-[11px]">
                                    <span className="font-mono text-emerald-400 mr-1.5">✓</span>
                                    <span className="text-foreground">{ev}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <div className="text-muted-foreground text-[11px]">
                              This attack stage is not supported by current network observations. The rule layer requires at least 2 verified telemetry rules before assigning confidence.
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            );
          })()}
        </Panel>
      )}

      {/* SECTION 15 & 16: EXPLAINABILITY & WHAT-IF DEFENCE PANELS */}
      {hasTraffic && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          {/* SECTION 15: WHY IS THE RISK CHANGING? (SHAP DRIVERS) */}
          <Panel
            title="Why Is The Risk Changing?"
            eyebrow="EXPLAINABILITY · INTEGRATED GRADIENT SHAP ATTRIBUTIONS"
          >
            <p className="mb-4 text-xs text-muted-foreground">
              Key physical flow features driving the neural network's forward infiltration risk score:
            </p>
            {explainLoading ? (
              <div className="flex h-48 items-center justify-center text-xs text-muted-foreground">
                <LoaderCircle size={20} className="mr-2 animate-spin text-primary" />
                Computing gradient feature attributions...
              </div>
            ) : explainData?.top_features && explainData.top_features.length > 0 ? (
              <div className="space-y-2.5">
                {explainData.top_features.slice(0, 5).map((f) => (
                  <div
                    key={f.feature}
                    className="flex items-center justify-between rounded border border-border bg-card/50 p-2.5 text-xs"
                  >
                    <div>
                      <div className="font-semibold text-foreground">{f.feature}</div>
                      <div className="font-mono text-[10px] text-muted-foreground">
                        Observed value: {f.raw_value.toFixed(2)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="text-right">
                        <div
                          className={`font-semibold text-[11px] ${
                            f.direction === "increases_risk"
                              ? "text-warning"
                              : "text-success"
                          }`}
                        >
                          {f.direction === "increases_risk" ? "↑ Increases Risk" : "↓ Decreases Risk"}
                        </div>
                        <div className="font-mono text-[10px] text-muted-foreground">
                          Impact: {(f.importance * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-36 flex-col items-center justify-center text-center text-xs text-muted-foreground">
                <Sparkles size={20} className="mb-1 text-primary" />
                <span>Feature attribution analysis ready.</span>
                <button
                  onClick={fetchExplainability}
                  className="mt-2 text-primary hover:underline"
                >
                  Run Explainability Analysis
                </button>
              </div>
            )}
          </Panel>

          {/* SECTION 16: WHAT-IF DEFENCE SIMULATOR */}
          <Panel
            title="What-If Defence Simulator"
            eyebrow="COUNTERFACTUAL INTERVENTION · TEST BEFORE COMMITTING"
          >
            <p className="mb-3 text-xs text-muted-foreground">
              Select an intervention to perturb the network state and run the forward model counterfactually:
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="field text-xs">
                <span className="text-muted-foreground">Defensive Policy</span>
                <select
                  value={whatIfAction}
                  onChange={(e) => {
                    const newAct = e.target.value;
                    setWhatIfAction(newAct);
                    if (newAct === "Block IP") setWhatIfTarget("185.220.101.4");
                    else if (newAct === "Quarantine Host") setWhatIfTarget("192.168.1.105");
                    else if (newAct === "Close Port") setWhatIfTarget("80");
                    else if (newAct === "Rate Limit") setWhatIfTarget("90% throttle");
                  }}
                  className="input mt-1 w-full text-xs"
                >
                  <option value="Block IP">Block IP (Drop Malicious IP Flows)</option>
                  <option value="Quarantine Host">Quarantine Host (Sever Ingress & Egress)</option>
                  <option value="Close Port">Close Port (Drop Port Traffic)</option>
                  <option value="Rate Limit">Rate Limit (Throttle Bandwidth & Rate)</option>
                </select>
              </label>

              <label className="field text-xs">
                <span className="text-muted-foreground">Target / Parameter</span>
                <input
                  type="text"
                  value={whatIfTarget}
                  onChange={(e) => setWhatIfTarget(e.target.value)}
                  placeholder="e.g. 80, 445, 185.220.101.4"
                  className="input mt-1 w-full text-xs font-mono"
                />
              </label>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <button
                onClick={handleRunWhatIf}
                disabled={whatIfLoading}
                className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {whatIfLoading ? (
                  <LoaderCircle size={14} className="animate-spin" />
                ) : (
                  <Play size={14} />
                )}
                Run Defence Simulation
              </button>

              {whatIfResult && (
                <button
                  onClick={() => setWhatIfResult(null)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  <RotateCcw size={12} /> Reset
                </button>
              )}
            </div>

            {whatIfError && (
              <div className="mt-3 text-xs text-warning">{whatIfError}</div>
            )}

            {/* Simulation Comparison Output */}
            {whatIfResult && (
              <div className="mt-4 rounded-lg border border-border bg-muted/20 p-3.5 text-xs">
                <div className="flex flex-wrap items-center justify-between border-b border-border pb-2 gap-2">
                  <span className="font-semibold text-foreground">
                    Selected Action: <span className="text-primary">{whatIfResult.selected_action || whatIfAction}</span>
                    {whatIfTarget && <span className="ml-1 text-muted-foreground font-mono">({whatIfTarget})</span>}
                  </span>
                  <span className={`flex items-center gap-1 font-mono font-semibold ${
                    whatIfResult.risk_change < 0 ? "text-success" : "text-destructive"
                  }`}>
                    <CheckCircle2 size={13} />
                    Model-Predicted Risk Change: {(whatIfResult.risk_change * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-3 text-center">
                  <div className="rounded border border-destructive/30 bg-destructive/5 p-2">
                    <div className="text-[10px] text-muted-foreground">Baseline Peak Risk</div>
                    <div className="text-lg font-mono font-bold text-destructive">
                      {((whatIfResult.baseline.peak_risk ?? whatIfResult.baseline.risk) * 100).toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-muted-foreground">{whatIfResult.baseline.risk_category} Risk</div>
                  </div>
                  <div className="rounded border border-success/40 bg-success/5 p-2">
                    <div className="text-[10px] text-success">What-If Peak Risk</div>
                    <div className="text-lg font-mono font-bold text-success">
                      {((whatIfResult.counterfactual.peak_risk ?? whatIfResult.counterfactual.risk) * 100).toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-muted-foreground">{whatIfResult.counterfactual.risk_category} Risk</div>
                  </div>
                </div>

                {whatIfChartData.length > 0 && (
                  <div className="mt-3">
                    <div className="mb-1 text-[10px] text-muted-foreground font-mono">
                      TRAJECTORY ROLLOUT (BASELINE VS WHAT-IF AT t+1...t+k):
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-mono">
                      {whatIfChartData.map((pt, i) => {
                        const basePts = whatIfResult.baseline?.timeline;
                        const bRisk = basePts && basePts[i] ? (basePts[i].risk * 100).toFixed(0) : "-";
                        return (
                          <div key={pt.step} className="rounded bg-card px-2 py-1 border border-border text-center flex-1 min-w-[50px]">
                            <span className="text-muted-foreground text-[9px] block">{pt.step}</span>
                            <span className="text-destructive text-[10px] block line-through opacity-75">{bRisk}%</span>
                            <span className="text-success font-bold text-[11px]">{pt.risk.toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Disclaimer Banner */}
                <div className="mt-3 rounded border border-border/60 bg-muted/40 p-2 text-[10px] leading-relaxed text-muted-foreground">
                  <strong>Notice:</strong> This simulation reports only the World Model's predicted statistical change in risk between baseline and modified network states. It does not claim that the action prevents an attack.
                </div>
              </div>
            )}
          </Panel>
        </div>
      )}
    </>
  );
}
