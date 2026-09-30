import { useState, useMemo, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Globe,
  Shield,
  Server,
  Database,
  Laptop,
  AlertTriangle,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  ChevronRight,
  ChevronLeft,
  Activity,
  Layers,
  Upload,
  Radio,
  Eye,
  Crosshair,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/common/Button";
import type { ForecastTrajectoryStep, ForecastTimelineItem, MitreMapping } from "@/types";

interface FutureTrajectoryVisualizerProps {
  timeline?: ForecastTimelineItem[];
  trajectory?: ForecastTrajectoryStep[];
  mitreMapping?: MitreMapping | null | undefined;
  hasTraffic: boolean;
  onUploadClick?: (() => void) | undefined;
}

interface StepIntelligence {
  step: number;
  horizon_label: string;
  status: "CURRENT" | "FORECAST";
  risk_probability: number;
  risk_percent: number;
  risk_level: string;
  attack_stage: string;
  network_state: string;
  key_changes: string;
  is_synthesized: boolean;
  flow_pkts_s: number;
  flow_bytes_s: number;
}

export function FutureTrajectoryVisualizer({
  timeline = [],
  trajectory = [],
  mitreMapping,
  hasTraffic,
  onUploadClick,
}: FutureTrajectoryVisualizerProps) {
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);
  const [hoveredStepIdx, setHoveredStepIdx] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [previewMode, setPreviewMode] = useState<boolean>(false);
  const playTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Canonical MITRE stages across 6 steps
  const stageByStep: Record<number, { stage: string; technique: string; state: string; changes: string }> = {
    0: {
      stage: "Reconnaissance",
      technique: "T1595 · Active Scanning",
      state: "Perimeter ingress probing, SYN scans on ports 80/443/445",
      changes: "Observed baseline traffic from external IP 185.220.101.4",
    },
    1: {
      stage: "Initial Access",
      technique: "T1190 · Exploit Public-Facing App",
      state: "HTTP anomalous payload transmission targeting DMZ Web Server",
      changes: "+65% packet rate increase, asymmetric request-to-response ratio",
    },
    2: {
      stage: "Execution & Persistence",
      technique: "T1059 · Command and Scripting",
      state: "DMZ web server process spawn, reverse shell connection opened",
      changes: "Outbound ephemeral socket initiated, shell spawned on port 4444",
    },
    3: {
      stage: "Lateral Movement",
      technique: "T1021 · Remote Services / SMB",
      state: "Pivot from DMZ across internal subnet 10.0.1.0/24 towards database",
      changes: "SMB port 445 connection burst, internal credential spraying",
    },
    4: {
      stage: "Command & Control",
      technique: "T1071 · Application Layer Protocol",
      state: "Persistent encrypted C2 beaconing on port 8443 at 30s intervals",
      changes: "Jittered heartbeat beaconing established, entropy elevation",
    },
    5: {
      stage: "Exfiltration",
      technique: "T1041 · Exfiltration Over C2",
      state: "Bulk staging and outbound data transfer from database core",
      changes: "Peak threat: +380% outbound byte volume transfer to egress gateway",
    },
  };

  // Normalize trajectory steps
  const steps: StepIntelligence[] = useMemo(() => {
    function getStageInfo(idx: number): { stage: string; technique: string; state: string; changes: string } {
      return stageByStep[idx] || {
        stage: "Reconnaissance",
        technique: "T1595 · Active Scanning",
        state: "Perimeter ingress probing, SYN scans on ports 80/443/445",
        changes: "Observed baseline traffic from external IP 185.220.101.4",
      };
    }

    if (hasTraffic && trajectory && trajectory.length > 0) {
      return trajectory.slice(0, 6).map((item, idx) => {
        const stageInfo = getStageInfo(idx);
        const riskPct = item.risk_percent ?? (item.risk_probability * 100);
        return {
          step: item.step ?? idx,
          horizon_label: idx === 0 ? "NOW" : `T+${idx}`,
          status: (idx === 0 ? "CURRENT" : "FORECAST") as "CURRENT" | "FORECAST",
          risk_probability: item.risk_probability,
          risk_percent: riskPct,
          risk_level: item.risk_level || (riskPct >= 75 ? "CRITICAL" : riskPct >= 50 ? "HIGH" : riskPct >= 25 ? "MEDIUM" : "LOW"),
          attack_stage: idx === 0 && mitreMapping?.stage ? mitreMapping.stage : stageInfo.stage,
          network_state: stageInfo.state,
          key_changes: item.description || stageInfo.changes,
          is_synthesized: idx > 0,
          flow_pkts_s: Number(item.key_features?.["Flow Pkts/s"] || (20 + idx * 12)),
          flow_bytes_s: Number(item.key_features?.["Flow Byts/s"] || (500 + idx * 600)),
        };
      });
    }

    if (hasTraffic && timeline && timeline.length > 0) {
      return timeline.slice(0, 6).map((item, idx) => {
        const stageInfo = getStageInfo(idx);
        const riskPct = item.risk > 1 ? item.risk : item.risk * 100;
        return {
          step: item.step ?? idx,
          horizon_label: idx === 0 ? "NOW" : `T+${idx}`,
          status: (idx === 0 ? "CURRENT" : "FORECAST") as "CURRENT" | "FORECAST",
          risk_probability: item.risk > 1 ? item.risk / 100 : item.risk,
          risk_percent: riskPct,
          risk_level: item.risk_category || (riskPct >= 75 ? "CRITICAL" : riskPct >= 50 ? "HIGH" : riskPct >= 25 ? "MEDIUM" : "LOW"),
          attack_stage: idx === 0 && mitreMapping?.stage ? mitreMapping.stage : stageInfo.stage,
          network_state: stageInfo.state,
          key_changes: item.description || stageInfo.changes,
          is_synthesized: idx > 0,
          flow_pkts_s: Number(item.key_features?.["Flow Pkts/s"] || (20 + idx * 12)),
          flow_bytes_s: Number(item.key_features?.["Flow Byts/s"] || (500 + idx * 600)),
        };
      });
    }

    // High fidelity preview baseline
    return [0, 1, 2, 3, 4, 5].map((idx) => {
      const stageInfo = getStageInfo(idx);
      const risks = [14, 32, 58, 74, 81, 86];
      const riskPct = risks[idx] ?? 14;
      return {
        step: idx,
        horizon_label: idx === 0 ? "NOW" : `T+${idx}`,
        status: (idx === 0 ? "CURRENT" : "FORECAST") as "CURRENT" | "FORECAST",
        risk_probability: riskPct / 100,
        risk_percent: riskPct,
        risk_level: riskPct >= 75 ? "CRITICAL" : riskPct >= 50 ? "HIGH" : riskPct >= 25 ? "MEDIUM" : "LOW",
        attack_stage: stageInfo.stage,
        network_state: stageInfo.state,
        key_changes: stageInfo.changes,
        is_synthesized: idx > 0,
        flow_pkts_s: 14.2 + idx * 15,
        flow_bytes_s: 420 + idx * 900,
      };
    });
  }, [hasTraffic, trajectory, timeline, mitreMapping]);

  // Autoplay through steps
  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = setInterval(() => {
        setSelectedStepIdx((prev) => (prev + 1) % steps.length);
      }, 2400);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, steps.length]);

  const fallbackStep: StepIntelligence = steps[0] || {
    step: 0,
    horizon_label: "NOW",
    status: "CURRENT",
    risk_probability: 0.1,
    risk_percent: 10,
    risk_level: "LOW",
    attack_stage: "Reconnaissance",
    network_state: "Normal baseline",
    key_changes: "Baseline ingress telemetry",
    is_synthesized: false,
    flow_pkts_s: 14,
    flow_bytes_s: 400,
  };
  const activeIdx = hoveredStepIdx !== null ? hoveredStepIdx : selectedStepIdx;
  const activeStep: StepIntelligence = steps[Math.min(activeIdx, Math.max(0, steps.length - 1))] || fallbackStep;
  const activeRisk = activeStep.risk_percent;
  const isElevated = activeRisk >= 50;
  const isCritical = activeRisk >= 75;

  // If no traffic and not in preview mode, render the required empty state
  if (!hasTraffic && !previewMode) {
    return (
      <div className="topology-container p-6 md:p-8">
        <div className="flex flex-col items-center justify-center py-12 px-4 text-center max-w-xl mx-auto">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/40 bg-primary/10 text-primary shadow-[0_0_20px_rgba(6,182,212,0.2)] mb-4">
            <Radio size={28} className="animate-pulse" />
          </div>

          <span className="eyebrow text-primary">PREDICTIVE DIGITAL TWIN STANDBY</span>
          <h3 className="mt-2 text-xl md:text-2xl font-bold tracking-tight text-foreground">
            NO NETWORK STATE LOADED
          </h3>
          <p className="mt-2 text-xs md:text-sm text-muted-foreground leading-relaxed">
            Upload traffic telemetry to initialize the predictive network model and generate the
            autoregressive future trajectory across <span className="font-mono text-primary">NOW → T+1..T+5</span>.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            <Button
              onClick={onUploadClick}
              className="shadow-sm font-semibold text-xs px-5 py-2.5"
            >
              <Upload size={14} />
              UPLOAD TRAFFIC
            </Button>

            <Button
              variant="outline"
              onClick={() => setPreviewMode(true)}
              className="font-mono text-xs text-muted-foreground hover:text-foreground"
            >
              <Eye size={14} className="text-primary" />
              Preview Model Trajectory
            </Button>
          </div>

          <div className="mt-8 grid grid-cols-3 gap-3 w-full border-t border-border/50 pt-5 text-left font-mono text-[10px]">
            <div className="rounded border border-border/40 bg-secondary/30 p-2.5">
              <span className="text-muted-foreground block">MODEL ENGINE</span>
              <span className="text-foreground font-semibold mt-0.5 block">51-D LSTM World Model</span>
            </div>
            <div className="rounded border border-border/40 bg-secondary/30 p-2.5">
              <span className="text-muted-foreground block">HORIZON DEPTH</span>
              <span className="text-foreground font-semibold mt-0.5 block">6-Step Autoregressive</span>
            </div>
            <div className="rounded border border-border/40 bg-secondary/30 p-2.5">
              <span className="text-muted-foreground block">EXPLAINABILITY</span>
              <span className="text-foreground font-semibold mt-0.5 block">Deep SHAP Gradients</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="topology-container p-4 md:p-6 transition-all">
      {/* Visualizer Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3.5 mb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/30 shadow-[0_0_12px_rgba(6,182,212,0.15)]">
            <Layers size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="eyebrow text-primary font-bold">SIGNATURE TRAJECTORY MODEL</span>
              <span className="text-muted-foreground text-xs">·</span>
              <span className="font-mono text-[10px] text-muted-foreground">K-STEP LOOKAHEAD</span>
            </div>
            <h3 className="text-base font-bold tracking-wide text-foreground">
              Future Network Trajectory & Digital Twin
            </h3>
          </div>
        </div>

        {/* Status Pill & Scrubber Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-1 rounded-md border border-border bg-secondary/60 p-1">
            <button
              onClick={() => setSelectedStepIdx((prev) => Math.max(0, prev - 1))}
              disabled={selectedStepIdx === 0}
              className="h-7 w-7 rounded flex items-center justify-center text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              title="Previous Step"
              aria-label="Previous step"
            >
              <ChevronLeft size={14} />
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="h-7 px-2.5 rounded bg-primary/15 hover:bg-primary/25 text-primary flex items-center gap-1 text-[11px] font-mono font-semibold transition-colors"
              title={isPlaying ? "Pause Timeline Rollout" : "Play Timeline Rollout"}
            >
              {isPlaying ? <Pause size={12} /> : <Play size={12} />}
              <span>{isPlaying ? "PAUSE" : "PLAY"}</span>
            </button>

            <button
              onClick={() => setSelectedStepIdx((prev) => Math.min(steps.length - 1, prev + 1))}
              disabled={selectedStepIdx === steps.length - 1}
              className="h-7 w-7 rounded flex items-center justify-center text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              title="Next Step"
              aria-label="Next step"
            >
              <ChevronRight size={14} />
            </button>
          </div>

          <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-[10px] font-semibold text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-ping" />
            {activeStep.is_synthesized
              ? "Simulation / Model Visualization"
              : hasTraffic
              ? "Physical Ingress Telemetry"
              : "Simulation / Model Preview"}
          </span>

          {!hasTraffic && previewMode && (
            <Button
              variant="outline"
              size="sm"
              onClick={onUploadClick}
              className="font-mono text-xs h-7"
            >
              <Upload size={12} />
              Upload Real CSV
            </Button>
          )}
        </div>
      </div>

      {/* Trajectory Step Scrubber Bar (NOW -> T+1 -> T+2 -> T+3 -> T+4 -> T+5) */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1.5 text-[10px] font-mono text-muted-foreground">
          <span className="tracking-wider uppercase">AUTOREGRESSIVE TEMPORAL ROLLOUT</span>
          <span className="text-foreground">
            ACTIVE HORIZON: <b className="text-primary">{activeStep.horizon_label}</b>
          </span>
        </div>

        <div className="grid grid-cols-6 gap-2">
          {steps.map((st, idx) => {
            const isSelected = idx === selectedStepIdx;
            const isHovered = idx === hoveredStepIdx;
            const riskColor =
              st.risk_percent >= 75
                ? "text-destructive"
                : st.risk_percent >= 50
                ? "text-warning"
                : "text-emerald-400";
            const borderColor =
              isSelected || isHovered
                ? "border-primary bg-primary/15 shadow-[0_0_12px_rgba(6,182,212,0.2)] ring-1 ring-primary"
                : "border-border/70 bg-card/60 hover:border-primary/50 hover:bg-card";

            return (
              <button
                key={st.step}
                onClick={() => setSelectedStepIdx(idx)}
                onMouseEnter={() => setHoveredStepIdx(idx)}
                onMouseLeave={() => setHoveredStepIdx(null)}
                className={`flex flex-col items-center justify-between rounded-lg border p-2.5 text-center transition-all cursor-pointer ${borderColor}`}
              >
                <div className="flex w-full items-center justify-between font-mono text-[10px] text-muted-foreground">
                  <span className="font-bold">{st.horizon_label}</span>
                  {idx === 0 ? (
                    <span className="rounded bg-primary/20 px-1 py-0.2 text-[8.5px] text-primary font-bold">
                      {hasTraffic ? "OBSERVED" : "BASELINE"}
                    </span>
                  ) : (
                    <span className="text-[9px] opacity-70">S(t+{idx})</span>
                  )}
                </div>

                <div className={`my-1 text-base font-bold font-mono ${riskColor}`}>
                  {st.risk_percent.toFixed(0)}%
                </div>

                <div className="w-full flex items-center justify-between font-mono text-[9px] text-muted-foreground">
                  <span className="truncate uppercase">{st.risk_level}</span>
                  {(isSelected || isHovered) && (
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Interactive Topology Graph SVG - Dark Navy Centerpiece Canvas */}
      <div className="relative h-[330px] md:h-[370px] w-full rounded-xl border border-[#0B1F3A] bg-[#061426] overflow-hidden shadow-2xl">
        {/* SVG Network Topology Map */}
        <svg
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="none"
          viewBox="0 0 900 370"
          aria-hidden="true"
        >
          <defs>
            {/* Grid Pattern */}
            <pattern id="networldGrid" width="45" height="45" patternUnits="userSpaceOnUse">
              <path
                d="M 45 0 L 0 0 0 45"
                fill="none"
                stroke="#12345A"
                strokeWidth="0.7"
                strokeOpacity="0.4"
              />
            </pattern>
            {/* Flow Gradients */}
            <linearGradient id="edgeThreatGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#dc2626" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.7" />
            </linearGradient>
            <linearGradient id="edgeNormalGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#16a34a" stopOpacity="0.6" />
            </linearGradient>
          </defs>

          {/* Background Grid */}
          <rect width="900" height="370" fill="url(#networldGrid)" />

          {/* Connection 1: External (130, 185) -> Gateway (310, 185) */}
          <line
            x1="140"
            y1="185"
            x2="290"
            y2="185"
            stroke={isElevated ? "url(#edgeThreatGrad)" : "url(#edgeNormalGrad)"}
            strokeWidth={isElevated ? "2.5" : "1.5"}
            strokeDasharray={isElevated ? "5 3" : "none"}
          />
          {/* Connection 2: Gateway (310, 185) -> DMZ Web Server (510, 105) */}
          <line
            x1="330"
            y1="175"
            x2="490"
            y2="115"
            stroke={isElevated ? "#dc2626" : "#06b6d4"}
            strokeWidth={isElevated ? "2.5" : "1.5"}
            strokeDasharray={isElevated ? "4 3" : "none"}
          />
          {/* Connection 3: Gateway (310, 185) -> Workstation (510, 265) */}
          <line
            x1="330"
            y1="195"
            x2="490"
            y2="255"
            stroke="#1e3a5f"
            strokeWidth="1.5"
            strokeOpacity="0.8"
          />
          {/* Connection 4: DMZ Web Server (510, 105) -> Core Database (730, 185) */}
          <line
            x1="530"
            y1="115"
            x2="710"
            y2="175"
            stroke={isCritical ? "#dc2626" : "#1e3a5f"}
            strokeWidth={isCritical ? "2.5" : "1.5"}
            strokeDasharray={isCritical ? "4 3" : "none"}
          />
          {/* Connection 5: Workstation (510, 265) -> Core Database (730, 185) */}
          <line
            x1="530"
            y1="255"
            x2="710"
            y2="195"
            stroke="#1e3a5f"
            strokeWidth="1.5"
            strokeOpacity="0.6"
          />

          {/* Animated Flow Packets traveling along connections */}
          <circle r={isElevated ? "4.5" : "3.5"} fill={isElevated ? "#dc2626" : "#06b6d4"}>
            <animateMotion
              path="M 140 185 L 290 185"
              dur={isElevated ? "1.4s" : "2.8s"}
              repeatCount="indefinite"
            />
          </circle>
          <circle r={isElevated ? "4.5" : "3.5"} fill={isElevated ? "#dc2626" : "#06b6d4"}>
            <animateMotion
              path="M 330 175 L 490 115"
              dur={isElevated ? "1.2s" : "2.4s"}
              repeatCount="indefinite"
            />
          </circle>
          <circle r="3" fill="#06b6d4" opacity="0.7">
            <animateMotion
              path="M 330 195 L 490 255"
              dur="3s"
              repeatCount="indefinite"
            />
          </circle>
          {isCritical && (
            <circle r="4.5" fill="#dc2626">
              <animateMotion
                path="M 530 115 L 710 175"
                dur="1s"
                repeatCount="indefinite"
              />
            </circle>
          )}
        </svg>

        {/* Node 1: External Adversary Source */}
        <div className="absolute left-[15%] top-[50%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
          <div
            className={`relative flex h-12 w-12 items-center justify-center rounded-xl border p-2 transition-all ${
              isElevated
                ? "border-red-500 bg-red-500/20 text-red-400 shadow-lg shadow-red-500/30 ring-1 ring-red-500"
                : "border-[#12345A] bg-[#0B1F3A] text-white shadow-md"
            }`}
          >
            <Globe size={22} />
            {isElevated && (
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
              </span>
            )}
          </div>
          <span className="mt-1.5 font-mono text-[10.5px] font-semibold text-white">
            External Ingress
          </span>
          <span className="font-mono text-[9px] text-slate-400">185.220.101.4</span>
        </div>

        {/* Node 2: Perimeter Gateway / Firewall */}
        <div className="absolute left-[34%] top-[50%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/50 bg-[#0B1F3A] text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.25)]">
            <Shield size={22} />
          </div>
          <span className="mt-1.5 font-mono text-[10.5px] font-semibold text-white">
            Perimeter Gateway
          </span>
          <span className="font-mono text-[9px] text-slate-400">10.0.0.1 (FW)</span>
        </div>

        {/* Node 3: DMZ Web Server (Port 80/443) */}
        <div className="absolute left-[58%] top-[29%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
          <div
            className={`flex h-12 w-12 items-center justify-center rounded-xl border p-2 transition-all ${
              isElevated
                ? "border-amber-500 bg-amber-500/20 text-amber-400 shadow-md shadow-amber-500/20 ring-1 ring-amber-500"
                : "border-[#12345A] bg-[#0B1F3A] text-white shadow-md"
            }`}
          >
            <Server size={22} />
          </div>
          <span className="mt-1.5 font-mono text-[10.5px] font-semibold text-white">
            DMZ Web Server
          </span>
          <span className="font-mono text-[9px] text-slate-400">10.0.0.5 · Port 80</span>
        </div>

        {/* Node 4: Workstation Host */}
        <div className="absolute left-[58%] top-[71%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#12345A] bg-[#0B1F3A] text-slate-300 p-2 shadow-md">
            <Laptop size={22} />
          </div>
          <span className="mt-1.5 font-mono text-[10.5px] font-semibold text-white">
            Corp Workstation
          </span>
          <span className="font-mono text-[9px] text-slate-400">10.0.1.14</span>
        </div>

        {/* Node 5: Core Database / File Server */}
        <div className="absolute left-[82%] top-[50%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
          <div
            className={`flex h-12 w-12 items-center justify-center rounded-xl border p-2 transition-all ${
              isCritical
                ? "border-red-500 bg-red-500/20 text-red-400 shadow-lg shadow-red-500/30 ring-1 ring-red-500"
                : "border-[#12345A] bg-[#0B1F3A] text-white shadow-md"
            }`}
          >
            <Database size={22} />
          </div>
          <span className="mt-1.5 font-mono text-[10.5px] font-semibold text-white">
            Database Core
          </span>
          <span className="font-mono text-[9px] text-slate-400">10.0.2.100 · Port 445</span>
        </div>
      </div>

      {/* Active Horizon Telemetry HUD Strip (4 Core Forecasting Indicators - Positioned Cleanly Underneath Canvas) */}
      <div className="mt-3.5 rounded-xl border border-[#CBD5E1] dark:border-[#12345A] bg-white dark:bg-[#061426] p-3 md:p-3.5 shadow-sm">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-xs">
          {/* Field 1: PREDICTED RISK */}
          <div className="rounded-lg border border-[#CBD5E1] dark:border-[#12345A] bg-slate-50/80 dark:bg-[#0B1F3A] p-3 flex flex-col justify-between">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-sans font-bold tracking-wider block">
              PREDICTED RISK
            </span>
            <div className="flex items-center gap-2 mt-1.5">
              <span
                className={`text-xl font-bold font-mono ${
                  activeRisk >= 75
                    ? "text-red-500 dark:text-red-400"
                    : activeRisk >= 50
                    ? "text-amber-500 dark:text-amber-400"
                    : "text-emerald-500 dark:text-emerald-400"
                }`}
              >
                {activeRisk.toFixed(1)}%
              </span>
              <span
                className={`rounded px-1.5 py-0.5 text-[9px] font-bold ${
                  activeRisk >= 75
                    ? "bg-red-500/15 text-red-500 dark:text-red-400"
                    : activeRisk >= 50
                    ? "bg-amber-500/15 text-amber-500 dark:text-amber-400"
                    : "bg-emerald-500/15 text-emerald-500 dark:text-emerald-400"
                }`}
              >
                {activeStep.risk_level}
              </span>
            </div>
          </div>

          {/* Field 2: ATTACK STAGE */}
          <div className="rounded-lg border border-[#CBD5E1] dark:border-[#12345A] bg-slate-50/80 dark:bg-[#0B1F3A] p-3 flex flex-col justify-between">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-sans font-bold tracking-wider block">
              ATTACK STAGE
            </span>
            <div className="mt-1.5 font-bold text-[#061426] dark:text-cyan-400 truncate text-[13px]" title={activeStep.attack_stage}>
              {activeStep.attack_stage}
            </div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
              Horizon Step: <span className="font-semibold text-[#0B1F3A] dark:text-white">{activeStep.horizon_label}</span>
            </div>
          </div>

          {/* Field 3: NETWORK STATE */}
          <div className="rounded-lg border border-[#CBD5E1] dark:border-[#12345A] bg-slate-50/80 dark:bg-[#0B1F3A] p-3 flex flex-col justify-between">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-sans font-bold tracking-wider block">
              NETWORK STATE
            </span>
            <div className="mt-1.5 text-[#061426] dark:text-white font-semibold truncate text-[11.5px]" title={activeStep.network_state}>
              {activeStep.network_state}
            </div>
            <div className="text-[10px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
              Rate: {activeStep.flow_pkts_s.toFixed(1)} pkts/s · {activeStep.flow_bytes_s} B/s
            </div>
          </div>

          {/* Field 4: KEY CHANGES */}
          <div className="rounded-lg border border-[#CBD5E1] dark:border-[#12345A] bg-slate-50/80 dark:bg-[#0B1F3A] p-3 flex flex-col justify-between">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-sans font-bold tracking-wider block">
              KEY CHANGES
            </span>
            <div className="mt-1.5 text-[#061426] dark:text-white text-[11px] leading-tight truncate" title={activeStep.key_changes}>
              {activeStep.key_changes}
            </div>
            <div className="text-[10px] text-cyan-600 dark:text-cyan-400 font-semibold truncate mt-0.5">
              Vector: {activeStep.is_synthesized ? "LSTM Rollout State" : "Ingress Baseline"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
