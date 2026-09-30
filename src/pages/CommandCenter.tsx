import { useEffect, useRef, useState, useMemo } from "react";
import { useNavigate } from "@tanstack/react-router";
import {
  ArrowRight,
  ShieldAlert,
  Sparkles,
  LoaderCircle,
  RefreshCw,
  Upload,
  AlertCircle,
  Database,
  Shield,
  CheckCircle2,
  ChevronRight,
  ChevronDown,
  Info,
  Sliders,
  TrendingUp,
  TrendingDown,
  Cpu,
  Radar,
  Activity,
  Check,
  Crosshair,
  ExternalLink,
  Layers,
  Terminal,
  ShieldCheck,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { Button } from "@/components/common/Button";
import { HelpTooltip } from "@/components/common/HelpTooltip";
import { FutureTrajectoryVisualizer } from "@/components/charts/FutureTrajectoryVisualizer";
import { RiskIntelligenceStrip } from "@/components/charts/RiskIntelligenceStrip";
import { ShapDivergingChart } from "@/components/charts/ShapDivergingChart";
import { WhatIfSplitScreen } from "@/components/charts/WhatIfSplitScreen";
import { runForecast, getExplanation } from "@/services/api";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import { useTheme } from "@/context/ThemeContext";
import type { AttackStageProgressionItem } from "@/types";

export default function CommandCenter() {
  const navigate = useNavigate();
  const session = useTrafficSession();
  const { theme } = useTheme();

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMitreStage, setSelectedMitreStage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const hasTraffic = Boolean(session.activeFilename || session.forecastResult);
  const forecastResult = session.forecastResult;
  const explainResult = session.explainResult;
  const isMitigated = session.postResponseRisk !== null;

  // Recompute forecast & explainability
  async function handleRecompute() {
    if (!session.activeFilename) return;
    setLoading(true);
    setError(null);
    try {
      const fc = await runForecast(10, undefined, session.activeFilename);
      let exp = null;
      try {
        exp = await getExplanation({
          top_n: 5,
          filename: session.activeFilename,
        });
      } catch {}

      trafficSession.setUploadData(
        session.uploadResult || {
          success: true,
          filename: session.activeFilename,
          rows: 0,
          columns: 0,
          missing_values: 0,
          model_features: 36,
          protocols: {},
          status: "ready",
          message: "Traffic session active",
        },
        fc,
        exp,
        session.originalName || undefined
      );
    } catch (err: any) {
      setError(err?.message || "Failed to recompute model forecast.");
    } finally {
      setLoading(false);
    }
  }

  // Handle CSV file upload
  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await trafficSession.analyzeTrafficFile(file);
    } catch (err: any) {
      setError(err?.message || "Failed to analyze uploaded traffic CSV.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  // Auto-compute when active traffic exists without forecast
  useEffect(() => {
    if (session.activeFilename && !session.forecastResult && !loading) {
      handleRecompute();
    }
  }, [session.activeFilename, session.forecastResult]);

  // MITRE progression computation
  const mitreMapping = forecastResult?.mitre_mapping;
  const progressionList = useMemo(() => {
    return mitreMapping?.progression || [];
  }, [mitreMapping]);

  const canonicalStages: Array<{
    num: string;
    key: string;
    label: string;
    techniqueDefault: string;
    desc: string;
  }> = [
    { num: "01", key: "Reconnaissance", label: "RECON", techniqueDefault: "T1595 · Active Scanning", desc: "Network footprinting and port probing" },
    { num: "02", key: "Initial Access", label: "INITIAL ACCESS", techniqueDefault: "T1190 · Exploit Public-Facing App", desc: "Perimeter service exploitation" },
    { num: "03", key: "Lateral Movement", label: "LATERAL MOVEMENT", techniqueDefault: "T1021 · Remote Services", desc: "Internal pivot across adjacent subnets" },
    { num: "04", key: "Command and Control", label: "C2", techniqueDefault: "T1071 · Application Layer Protocol", desc: "Beaconing to external command infrastructure" },
    { num: "05", key: "Exfiltration", label: "EXFILTRATION", techniqueDefault: "T1041 · Exfiltration Over C2 Channel", desc: "Unauthorized bulk outbound data transfer" },
  ];

  return (
    <div className="space-y-6">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={handleFileUpload}
      />

      {/* ========================================================================= */}
      {/* 1. TOP COMMAND AREA: NETWORK SECURITY COMMAND CENTER                      */}
      {/* ========================================================================= */}
      <div className="rounded-xl border border-[#0B1F3A] bg-gradient-to-r from-[#061426] via-[#0B1F3A] to-[#12345A] p-5 md:p-6 text-white shadow-xl relative overflow-hidden">
        {/* Subtle decorative glow */}
        <div className="pointer-events-none absolute -right-10 -top-10 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="pointer-events-none absolute right-1/4 -bottom-10 h-40 w-40 rounded-full bg-blue-500/10 blur-2xl" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-white/15 pb-5">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="inline-flex items-center gap-1.5 rounded bg-cyan-500/15 px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-wider text-cyan-400 border border-cyan-500/30">
                <Radar size={12} className="animate-spin-slow text-cyan-400" />
                CYBER COMMAND SYSTEM
              </span>
              <span className="text-slate-400 text-xs">·</span>
              <span className="font-mono text-xs text-slate-300 tracking-wide">
                ORBITAL OBSERVABILITY
              </span>
            </div>

            <h1 className="font-display text-2xl md:text-3xl font-extrabold tracking-tight text-white">
              NETWORK SECURITY COMMAND CENTER
            </h1>
            <p className="mt-1 text-xs md:text-sm text-slate-300 font-mono">
              Predictive Network Security ·{" "}
              <span className="text-cyan-400 font-semibold">
                Observe → Understand → Forecast → Explain → Simulate → Decide
              </span>
            </p>
          </div>

          {/* Quick Action Controls */}
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center justify-center gap-2 rounded-lg font-semibold text-xs h-9 px-4 uppercase tracking-wider bg-cyan-400 text-[#061426] hover:bg-cyan-300 transition-all shadow-[0_0_15px_rgba(6,182,212,0.35)] cursor-pointer disabled:opacity-50 font-mono"
            >
              {uploading ? (
                <LoaderCircle size={15} className="animate-spin" />
              ) : (
                <Upload size={15} />
              )}
              {uploading ? "Analyzing CSV..." : "Upload Traffic CSV"}
            </button>

            <button
              onClick={() => navigate({ to: "/what-if" })}
              className="inline-flex items-center justify-center gap-2 rounded-lg font-mono text-xs h-9 px-3.5 border border-white/25 bg-white/10 text-white hover:bg-white/20 transition-all cursor-pointer"
            >
              <Sliders size={14} className="text-cyan-400" /> Test What-If
            </button>

            {hasTraffic && (
              <>
                <button
                  onClick={handleRecompute}
                  disabled={loading}
                  className="inline-flex items-center justify-center gap-2 rounded-lg font-mono text-xs h-9 px-3.5 border border-white/25 bg-white/10 text-white hover:bg-white/20 transition-all cursor-pointer disabled:opacity-50"
                >
                  {loading ? (
                    <LoaderCircle size={14} className="animate-spin" />
                  ) : (
                    <RefreshCw size={14} />
                  )}
                  Recompute
                </button>

                <button
                  onClick={() => trafficSession.clearSession()}
                  className="inline-flex items-center justify-center gap-2 rounded-lg font-mono text-xs h-9 px-3.5 border border-red-500/40 bg-red-500/20 text-red-200 hover:bg-red-500/30 transition-all cursor-pointer"
                  title="Clear telemetry session"
                >
                  Clear
                </button>
              </>
            )}
          </div>
        </div>

        {/* COMPACT SYSTEM-STATUS STRIP */}
        <div className="relative z-10 mt-4 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex flex-wrap items-center gap-3 md:gap-5 text-slate-300 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-white">DATA SOURCE:</span>
              <span className="text-cyan-400 font-bold">
                {hasTraffic ? session.originalName || "LIVE TELEMETRY" : "CIC-IDS2017 BASELINE"}
              </span>
            </div>

            <div className="hidden sm:inline text-slate-500">·</div>

            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-white">MODEL STATUS:</span>
              <span className="text-emerald-400 font-bold flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                MULTI-HEAD LSTM (51-D UNIFIED)
              </span>
            </div>

            <div className="hidden md:inline text-slate-500">·</div>

            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-white">LAST ANALYSIS:</span>
              <span className="text-white">
                {hasTraffic ? "SYNCED" : "AWAITING TELEMETRY"}
              </span>
            </div>

            <div className="hidden lg:inline text-slate-500">·</div>

            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-white">FORECAST HORIZON:</span>
              <span className="text-cyan-400 font-bold">
                {forecastResult ? `T+${forecastResult.horizon || 5}` : "T+5"}
              </span>
            </div>

            <div className="hidden xl:inline text-slate-500">·</div>

            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-white">SYSTEM HEALTH:</span>
              <span className="text-emerald-400 font-bold">100% NOMINAL</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-bold ${
                hasTraffic
                  ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30"
                  : "bg-amber-500/10 text-amber-500 border border-amber-500/30"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  hasTraffic ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              {hasTraffic ? "TELEMETRY ACTIVE" : "STANDBY MODE"}
            </span>
          </div>
        </div>
      </div>

      {/* ERROR BANNER */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-destructive/40 bg-destructive/10 p-3.5 text-xs text-destructive">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* ========================================================================= */}
      {/* 2. NARRATIVE STEP: CURRENT STATE (RISK INTELLIGENCE STRIP)                */}
      {/* ========================================================================= */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="eyebrow text-primary">CURRENT STATE // OBSERVABILITY & RISK ENVELOPE</span>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground">
            STATE VECTOR S(NOW)
          </span>
        </div>
        <RiskIntelligenceStrip
          forecastResult={forecastResult}
          hasTraffic={hasTraffic}
        />
      </section>

      {/* ========================================================================= */}
      {/* 3. NARRATIVE STEP: FUTURE STATE (FUTURE NETWORK TRAJECTORY / DIGITAL TWIN)*/}
      {/* ========================================================================= */}
      <section className="space-y-2">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="eyebrow text-primary">FUTURE STATE // DIGITAL TWIN AUTOREGRESSIVE ROLLOUT</span>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground">
            NOW → T+1..T+5
          </span>
        </div>
        <FutureTrajectoryVisualizer
          timeline={forecastResult?.timeline || []}
          trajectory={forecastResult?.trajectory || []}
          mitreMapping={forecastResult?.mitre_mapping}
          hasTraffic={hasTraffic}
          onUploadClick={() => fileInputRef.current?.click()}
        />
      </section>

      {/* ========================================================================= */}
      {/* 4. NARRATIVE STEP: WHY? (ATTACK PROGRESSION & SHAP NEURAL EXPLAINABILITY) */}
      {/* ========================================================================= */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="eyebrow text-primary">WHY THIS THREAT? // ATTACK PROGRESSION & SHAP ATTRIBUTION</span>
          <span className="font-mono text-[10px] text-muted-foreground">
            EVIDENCE-BASED FORECAST REASONING
          </span>
        </div>

        {/* MITRE ATT&CK STAGE EVIDENCE PIPELINE */}
        <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="eyebrow">ATTACK STAGE PROGRESSION · MITRE ATT&CK EVIDENCE PIPELINE</span>
                <HelpTooltip
                  term="Attack Progression"
                  text="Deterministic verification of adversary intrusion phases using calibrated telemetry evidence rules. Only stages supported by observed flow and packet patterns are marked supported."
                />
              </div>
              <h3 className="text-base font-bold text-foreground mt-0.5">
                HORIZONTAL ATTACK PROGRESSION
              </h3>
            </div>

            <div className="font-mono text-xs text-muted-foreground flex items-center gap-2">
              <span>Current Stage:</span>
              <span className="font-bold text-primary">
                {hasTraffic && mitreMapping?.stage ? mitreMapping.stage : "AWAITING TELEMETRY"}
              </span>
              {hasTraffic && mitreMapping?.confidence_score !== undefined && (
                <span className="rounded bg-primary/10 border border-primary/30 px-2 py-0.5 text-[10px] text-primary font-bold">
                  {Math.round(mitreMapping.confidence_score * 100)}% CONFIDENCE
                </span>
              )}
            </div>
          </div>

          {/* 5-STAGE HORIZONTAL FLOW */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {canonicalStages.map((stageItem) => {
              const matchedItem = progressionList.find(
                (p) =>
                  p.stage.toLowerCase() === stageItem.key.toLowerCase() ||
                  p.stage.toLowerCase().includes(stageItem.label.toLowerCase()) ||
                  (stageItem.label === "C2" && p.stage.toLowerCase().includes("command"))
              );

              const isSupported = matchedItem ? matchedItem.supported : false;
              const isCurrent =
                hasTraffic &&
                mitreMapping?.stage &&
                (mitreMapping.stage.toLowerCase().includes(stageItem.label.toLowerCase()) ||
                  mitreMapping.stage.toLowerCase() === stageItem.key.toLowerCase() ||
                  (stageItem.label === "C2" && mitreMapping.stage.toLowerCase().includes("command")));

              const confidence = matchedItem
                ? matchedItem.confidence_percent
                : isCurrent && mitreMapping?.confidence_score !== undefined
                ? Math.round(mitreMapping.confidence_score * 100)
                : 0;

              const evidenceRules =
                matchedItem?.supporting_evidence ||
                (isCurrent ? mitreMapping?.evidence?.rules_matched || [] : []);

              const isSelected = selectedMitreStage === stageItem.key;

              return (
                <div
                  key={stageItem.num}
                  onClick={() => setSelectedMitreStage(isSelected ? null : stageItem.key)}
                  className={`relative cursor-pointer rounded-lg border p-3.5 transition-all ${
                    isCurrent && isSupported
                      ? "border-primary bg-primary/10 shadow-[0_0_15px_rgba(6,182,212,0.15)] ring-1 ring-primary"
                      : isSupported
                      ? "border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60"
                      : "border-border/40 bg-muted/10 opacity-60 hover:opacity-80"
                  }`}
                >
                  {/* Step Header */}
                  <div className="flex items-center justify-between text-[10px] font-mono">
                    <span className="text-muted-foreground font-bold">{stageItem.num}</span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[9px] font-bold ${
                        isSupported
                          ? "bg-emerald-500/15 text-emerald-500 border border-emerald-500/30"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {isSupported ? "SUPPORTED" : "UNSUPPORTED"}
                    </span>
                  </div>

                  {/* Stage Title */}
                  <div className="mt-2">
                    <b className="text-xs font-bold text-foreground block truncate">
                      {stageItem.key}
                    </b>
                    <span className="mt-0.5 block font-mono text-[10px] text-primary truncate">
                      {matchedItem?.technique_id || stageItem.techniqueDefault.split("·")[0]?.trim() || "T1595"}
                    </span>
                  </div>

                  {/* Current Stage Badge */}
                  {isCurrent && (
                    <div className="mt-2 inline-flex items-center gap-1 rounded bg-primary px-1.5 py-0.5 font-mono text-[8.5px] font-bold text-primary-foreground shadow-xs">
                      <Crosshair size={10} /> CURRENT PREDICTED STAGE
                    </div>
                  )}

                  {/* Confidence Bar */}
                  <div className="mt-3 border-t border-border/40 pt-2 text-[10px] font-mono">
                    <div className="flex justify-between items-center text-muted-foreground">
                      <span>Confidence:</span>
                      <span className={`font-bold ${isSupported ? "text-foreground" : "text-muted-foreground"}`}>
                        {confidence.toFixed(1)}%
                      </span>
                    </div>

                    {isSupported && (
                      <div className="mt-1 text-[9.5px] text-emerald-500 flex items-center gap-1">
                        <Check size={11} /> {evidenceRules.length} Verified Rule{evidenceRules.length === 1 ? "" : "s"}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* EXPANDABLE TELEMETRY EVIDENCE AUDIT DRAWER */}
          <AnimatePresence>
            {selectedMitreStage && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 border-t border-border/60 pt-3"
              >
                <div className="rounded-lg border border-border bg-card p-4 text-xs font-mono">
                  <div className="flex items-center justify-between border-b border-border/40 pb-2 mb-3">
                    <div className="flex items-center gap-2">
                      <Terminal size={14} className="text-primary" />
                      <span className="font-bold text-foreground">
                        TELEMETRY AUDIT DRAWER · {selectedMitreStage.toUpperCase()}
                      </span>
                    </div>
                    <button
                      onClick={() => setSelectedMitreStage(null)}
                      className="text-muted-foreground hover:text-foreground text-xs cursor-pointer"
                    >
                      ✕ Close
                    </button>
                  </div>

                  {(() => {
                    const item = progressionList.find(
                      (p) =>
                        p.stage.toLowerCase() === selectedMitreStage.toLowerCase() ||
                        (selectedMitreStage.includes("Command") && p.stage.toLowerCase().includes("command"))
                    );

                    if (item && item.supported && (item.supporting_evidence?.length ?? 0) > 0) {
                      return (
                        <div className="space-y-2">
                          <div className="text-primary font-semibold text-[11px]">
                            Verified Telemetry Rules ({item.rules_satisfied || item.supporting_evidence?.length || 0} satisfied):
                          </div>
                          <div className="space-y-1">
                            {(item.supporting_evidence || []).map((ev, i) => (
                              <div key={i} className="flex items-start gap-2 text-muted-foreground">
                                <span className="text-emerald-500 font-bold shrink-0">•</span>
                                <span className="text-foreground">{ev}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    }

                    return (
                      <div className="text-muted-foreground italic py-2">
                        No verified telemetry evidence detected for this stage in current observations.
                      </div>
                    );
                  })()}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* SHAP NEURAL EXPLAINABILITY: "WHY THIS ALERT?" */}
        <section>
          <ShapDivergingChart
            explainData={explainResult}
            predictedRisk={forecastResult ? forecastResult.highest_predicted_risk * 100 : null}
            isLoading={loading}
          />
        </section>
      </div>

      {/* ========================================================================= */}
      {/* 5. NARRATIVE STEP: WHAT-IF (COUNTERFACTUAL DEFENCE SIMULATION)             */}
      {/* ========================================================================= */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <span className="eyebrow text-primary">WHAT-IF // TEST DEFENSIVE COUNTERFACTUALS</span>
          <span className="font-mono text-[10px] text-muted-foreground">
            VIRTUAL TWIN INTERVENTION
          </span>
        </div>
        <WhatIfSplitScreen
          activeFilename={session.activeFilename}
          initialTimeline={forecastResult?.timeline || []}
          onSimulationComplete={(res) => {}}
        />
      </section>

      {/* ========================================================================= */}
      {/* 6. NARRATIVE STEP: RESPONSE (DEFENDER VERIFICATION & ACTIVE ENFORCEMENT)  */}
      {/* ========================================================================= */}
      <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border/70 pb-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="eyebrow text-primary">DEFENDER RESPONSE // POLICY VERIFICATION</span>
            </div>
            <h3 className="text-base font-bold text-foreground mt-0.5">
              ACTIVE ENFORCEMENT & MITIGATION AUDIT
            </h3>
          </div>

          <Button
            variant="outline"
            onClick={() => navigate({ to: "/response" })}
            className="font-mono text-xs"
          >
            Open Defender Response Hub <ArrowRight size={13} />
          </Button>
        </div>

        {isMitigated ? (
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-4 text-xs">
            <div className="flex items-center gap-3">
              <CheckCircle2 size={24} className="text-emerald-500 shrink-0" />
              <div>
                <b className="font-semibold text-foreground text-sm">
                  Active Mitigation Applied: {session.activeResponse?.action || "Policy Enforcement"}
                </b>
                <p className="mt-0.5 text-muted-foreground text-xs">
                  Threat risk quenched from{" "}
                  <span className="font-mono text-destructive font-bold">
                    {(session.preResponseRisk! * 100).toFixed(1)}%
                  </span>{" "}
                  down to{" "}
                  <span className="font-mono text-emerald-500 font-bold">
                    {(session.postResponseRisk! * 100).toFixed(1)}%
                  </span>{" "}
                  (Net safety shift:{" "}
                  <span className="font-mono font-bold text-primary">
                    {Math.abs(session.riskChangePts || 0)} pts
                  </span>
                  ). Telemetry Status: <strong>{session.verificationStatus}</strong>.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 rounded-lg border border-border/60 bg-secondary/30 p-4 text-xs font-mono">
            <div className="flex items-center gap-3">
              <ShieldCheck size={20} className="text-primary shrink-0" />
              <div>
                <span className="text-foreground font-semibold block">
                  Automated Defense Policy Available
                </span>
                <span className="text-muted-foreground text-[11px]">
                  Model recommends prophylactic port isolation or host rate-limiting before predicted attack step T+3.
                </span>
              </div>
            </div>

            <Button
              onClick={() => navigate({ to: "/response" })}
              size="sm"
              className="font-mono text-xs font-bold"
            >
              Review Defender Recommendations
            </Button>
          </div>
        )}
      </section>
    </div>
  );
}
