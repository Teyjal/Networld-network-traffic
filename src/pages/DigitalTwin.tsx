import { useState, useEffect, useMemo } from "react";
import {
  Globe,
  Shield,
  Server,
  Database,
  Laptop,
  HardDrive,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Search,
  Filter,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sliders,
  Layers,
  Radio,
  Eye,
  Info,
  Terminal,
} from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { Button } from "@/components/common/Button";
import { PageTitle, Panel, RiskBadge } from "@/components/common/Panel";
import { useTrafficSession } from "@/services/trafficSession";
import { getNetworkGraph, getTraffic } from "@/services/api";
import type { NetworkNode, TrafficFlow } from "@/types";

interface TwinNode {
  id: string;
  label: string;
  sublabel: string;
  ip: string;
  type: "external" | "gateway" | "web" | "workstation" | "database" | "storage";
  x: number; // percentage
  y: number; // percentage
  baseRisk: number; // percentage
  connections: number;
  ports: string;
  trafficVolume: string;
  status: "normal" | "suspicious" | "anomalous" | "compromised" | "isolated";
  stage: string;
  description: string;
}

const DEFAULT_TWIN_NODES: TwinNode[] = [
  {
    id: "external",
    label: "External Threat Vector",
    sublabel: "Adversary Ingress",
    ip: "185.220.101.4",
    type: "external",
    x: 12,
    y: 50,
    baseRisk: 88,
    connections: 18,
    ports: "80, 443, 445, 4444",
    trafficVolume: "48.2 pkts/s",
    status: "compromised",
    stage: "Active Infiltration",
    description: "External untrusted IP transmitting abnormal packet bursts matching exploit signatures.",
  },
  {
    id: "gateway",
    label: "Perimeter Gateway & IPS",
    sublabel: "Boundary Inspector",
    ip: "10.0.0.1 (FW)",
    type: "gateway",
    x: 32,
    y: 50,
    baseRisk: 34,
    connections: 42,
    ports: "Stateful Inspection",
    trafficVolume: "128.4 pkts/s",
    status: "normal",
    stage: "Filtering Ingress",
    description: "Next-gen edge gateway enforcing L3/L4 ACLs and TLS inspection policies.",
  },
  {
    id: "web",
    label: "DMZ Web Cluster",
    sublabel: "Public Ingress Target",
    ip: "10.0.0.5",
    type: "web",
    x: 55,
    y: 28,
    baseRisk: 76,
    connections: 24,
    ports: "80/TCP (HTTP), 443/TCP",
    trafficVolume: "84.0 pkts/s",
    status: "anomalous",
    stage: "Initial Access Target",
    description: "Reverse-proxy and web application server experiencing anomalous request rate elevation.",
  },
  {
    id: "workstation",
    label: "Engineering Workstation",
    sublabel: "Internal Subnet Client",
    ip: "10.0.1.14",
    type: "workstation",
    x: 55,
    y: 72,
    baseRisk: 22,
    connections: 9,
    ports: "Ephemeral Ports, 3389",
    trafficVolume: "14.5 pkts/s",
    status: "normal",
    stage: "Internal Client",
    description: "Developer client node running authorized corporate enterprise services.",
  },
  {
    id: "database",
    label: "Core Database Cluster",
    sublabel: "High-Value Asset",
    ip: "10.0.2.100",
    type: "database",
    x: 82,
    y: 35,
    baseRisk: 82,
    connections: 16,
    ports: "1433/TCP (SQL), 445/TCP",
    trafficVolume: "42.1 pkts/s",
    status: "suspicious",
    stage: "Targeted in Lateral Path",
    description: "Confidential database instance containing proprietary state telemetry and user accounts.",
  },
  {
    id: "storage",
    label: "Encrypted Backup Vault",
    sublabel: "Immutable Storage",
    ip: "10.0.2.150",
    type: "storage",
    x: 82,
    y: 65,
    baseRisk: 15,
    connections: 6,
    ports: "2049/TCP (NFS)",
    trafficVolume: "8.2 pkts/s",
    status: "normal",
    stage: "Target Asset",
    description: "Write-once read-many immutable backup system protected by secondary firewall.",
  },
];

const TWIN_LINKS: Array<{ source: string; target: string; isThreatPath: boolean }> = [
  { source: "external", target: "gateway", isThreatPath: true },
  { source: "gateway", target: "web", isThreatPath: true },
  { source: "gateway", target: "workstation", isThreatPath: false },
  { source: "web", target: "database", isThreatPath: true },
  { source: "workstation", target: "database", isThreatPath: false },
  { source: "database", target: "storage", isThreatPath: false },
];

export default function DigitalTwin() {
  const navigate = useNavigate();
  const session = useTrafficSession();

  const [nodes, setNodes] = useState<TwinNode[]>(DEFAULT_TWIN_NODES);
  const [selectedNode, setSelectedNode] = useState<TwinNode>(DEFAULT_TWIN_NODES[2]!);
  const [selectedHorizonStep, setSelectedHorizonStep] = useState<number>(0);
  const [filterMode, setFilterMode] = useState<"all" | "threat" | "benign">("all");
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [flows, setFlows] = useState<TrafficFlow[]>([]);

  const hasTraffic = Boolean(session.activeFilename || session.forecastResult);

  // Load flows if session is active
  useEffect(() => {
    if (session.activeFilename) {
      getTraffic(25).then((data) => setFlows(data));
    }
  }, [session.activeFilename]);

  // Adjust node risks dynamically as horizon steps increase (NOW -> T+5)
  const dynamicNodes = useMemo(() => {
    const horizonMultiplier = 1 + selectedHorizonStep * 0.12;
    return nodes.map((node) => {
      let risk = node.baseRisk;
      if (node.id === "web" || node.id === "database" || node.id === "external") {
        risk = Math.min(99, Math.round(node.baseRisk * horizonMultiplier));
      } else if (node.id === "workstation" && selectedHorizonStep >= 3) {
        risk = Math.min(65, Math.round(node.baseRisk + selectedHorizonStep * 8));
      }

      let status = node.status;
      if (risk >= 75) status = "compromised";
      else if (risk >= 50) status = "anomalous";
      else if (risk >= 30) status = "suspicious";
      else status = "normal";

      return {
        ...node,
        computedRisk: risk,
        dynamicStatus: status,
      };
    });
  }, [nodes, selectedHorizonStep]);

  const activeInspected: TwinNode & { computedRisk: number; dynamicStatus: string } =
    dynamicNodes.find((n) => n.id === selectedNode.id) ||
    dynamicNodes[0] || {
      ...DEFAULT_TWIN_NODES[0]!,
      computedRisk: DEFAULT_TWIN_NODES[0]!.baseRisk,
      dynamicStatus: DEFAULT_TWIN_NODES[0]!.status,
    };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-border/70 pb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="eyebrow text-primary">PREDICTIVE DIGITAL TWIN // FULL TOPOLOGY RUNTIME</span>
            <span className="text-muted-foreground text-xs">·</span>
            <span className="font-mono text-xs text-muted-foreground">REAL-TIME GRAPH STATE</span>
          </div>
          <h1 className="font-display text-2xl md:text-3xl font-extrabold tracking-tight text-foreground">
            Digital Twin Cyber Observability
          </h1>
          <p className="mt-1 text-xs md:text-sm text-muted-foreground font-mono">
            Explore topological state propagation, simulated attack paths, and prospective risk propagation across assets.
          </p>
        </div>

        {/* Global Action / Status */}
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-xs font-semibold text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-ping" />
            {hasTraffic ? "Physical Ingress Telemetry Linked" : "Simulation / Model Visualization"}
          </span>

          <Button
            variant="outline"
            onClick={() => navigate({ to: "/what-if" })}
            className="font-mono text-xs"
          >
            <Sliders size={13} className="text-primary" /> Test What-If on Twin
          </Button>
        </div>
      </div>

      {/* Main Grid: Left = Topology Canvas + Scrubber; Right = Node Inspector */}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        {/* TOPOLOGY CANVAS & INTERACTION SUITE */}
        <div className="space-y-4">
          <div className="topology-container p-4 md:p-6 shadow-sm">
            {/* Top Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3.5 mb-4">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold text-foreground uppercase tracking-wider">
                  TOPOLOGY MAP · CLICK ANY NODE TO INSPECT
                </span>
              </div>

              {/* Filter Segmented Control */}
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10.5px] text-muted-foreground hidden sm:inline">
                  Layer:
                </span>
                <div className="segmented">
                  <button
                    onClick={() => setFilterMode("all")}
                    className={filterMode === "all" ? "active" : ""}
                  >
                    All Flows
                  </button>
                  <button
                    onClick={() => setFilterMode("threat")}
                    className={filterMode === "threat" ? "active" : ""}
                  >
                    Threat Path
                  </button>
                  <button
                    onClick={() => setFilterMode("benign")}
                    className={filterMode === "benign" ? "active" : ""}
                  >
                    Benign Only
                  </button>
                </div>
              </div>
            </div>

            {/* Interactive SVG Network Graph */}
            <div className="relative h-[380px] md:h-[440px] w-full rounded-xl border border-border/80 bg-background/80 overflow-hidden shadow-inner">
              <svg
                className="absolute inset-0 h-full w-full"
                preserveAspectRatio="none"
                viewBox="0 0 100 100"
                aria-hidden="true"
              >
                <defs>
                  {/* Flow gradients */}
                  <linearGradient id="twinThreatGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.85" />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.75" />
                  </linearGradient>
                  <linearGradient id="twinBenignGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.5" />
                  </linearGradient>
                </defs>

                {/* Connection Lines */}
                {TWIN_LINKS.map((link, idx) => {
                  const src = dynamicNodes.find((n) => n.id === link.source);
                  const tgt = dynamicNodes.find((n) => n.id === link.target);
                  if (!src || !tgt) return null;

                  if (filterMode === "threat" && !link.isThreatPath) return null;
                  if (filterMode === "benign" && link.isThreatPath) return null;

                  const isHighThreat = link.isThreatPath && (activeInspected.computedRisk >= 50 || selectedHorizonStep >= 2);

                  return (
                    <g key={idx}>
                      <line
                        x1={src.x}
                        y1={src.y}
                        x2={tgt.x}
                        y2={tgt.y}
                        stroke={isHighThreat ? "url(#twinThreatGrad)" : "url(#twinBenignGrad)"}
                        strokeWidth={isHighThreat ? "0.65" : "0.4"}
                        strokeDasharray={isHighThreat ? "1.5 1" : "none"}
                      />

                      {/* Animated Packets */}
                      {isPlaying && (
                        <circle
                          r={isHighThreat ? "0.8" : "0.55"}
                          fill={isHighThreat ? "#f43f5e" : "#06b6d4"}
                        >
                          <animateMotion
                            path={`M ${src.x} ${src.y} L ${tgt.x} ${tgt.y}`}
                            dur={isHighThreat ? "1.4s" : "2.6s"}
                            repeatCount="indefinite"
                          />
                        </circle>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Render Nodes as Interactive Overlay Buttons */}
              {dynamicNodes.map((node) => {
                const isSelected = selectedNode.id === node.id;
                const riskColor =
                  node.computedRisk >= 75
                    ? "border-destructive bg-destructive/15 text-destructive shadow-lg shadow-destructive/25 ring-1 ring-destructive"
                    : node.computedRisk >= 50
                    ? "border-warning bg-warning/15 text-warning shadow-md shadow-warning/20 ring-1 ring-warning"
                    : node.computedRisk >= 30
                    ? "border-primary/60 bg-primary/10 text-primary"
                    : "border-border bg-card text-foreground";

                const Icon =
                  node.type === "external"
                    ? Globe
                    : node.type === "gateway"
                    ? Shield
                    : node.type === "web"
                    ? Server
                    : node.type === "workstation"
                    ? Laptop
                    : node.type === "database"
                    ? Database
                    : HardDrive;

                return (
                  <button
                    key={node.id}
                    onClick={() => setSelectedNode(node)}
                    style={{ left: `${node.x}%`, top: `${node.y}%` }}
                    className={`absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center cursor-pointer transition-all hover:scale-105 ${
                      isSelected ? "z-20 scale-110" : "z-10"
                    }`}
                  >
                    <div
                      className={`flex h-11 w-11 md:h-12 md:w-12 items-center justify-center rounded-xl border p-2 transition-all ${riskColor} ${
                        isSelected ? "ring-2 ring-primary shadow-xl" : ""
                      }`}
                    >
                      <Icon size={22} />
                      {node.computedRisk >= 75 && (
                        <span className="absolute -top-1 -right-1 flex h-3 w-3">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive opacity-75" />
                          <span className="relative inline-flex rounded-full h-3 w-3 bg-destructive" />
                        </span>
                      )}
                    </div>

                    <span className="mt-1.5 font-mono text-[10.5px] font-bold text-foreground text-center whitespace-nowrap bg-background/80 px-1.5 py-0.5 rounded border border-border/40 shadow-xs">
                      {node.label}
                    </span>
                    <span className="font-mono text-[9px] text-muted-foreground whitespace-nowrap">
                      {node.ip}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Trajectory Step Scrubber Strip */}
            <div className="mt-4 pt-3.5 border-t border-border/60">
              <div className="flex items-center justify-between mb-2 text-xs font-mono">
                <span className="text-muted-foreground">
                  SIMULATION HORIZON STEP: <strong className="text-primary">{selectedHorizonStep === 0 ? "NOW (Ingress)" : `T+${selectedHorizonStep}`}</strong>
                </span>
                <span className="text-[11px] text-muted-foreground">
                  Scrub to project digital twin state over time
                </span>
              </div>

              <div className="grid grid-cols-6 gap-2">
                {[0, 1, 2, 3, 4, 5].map((stepIdx) => {
                  const isCurrent = stepIdx === selectedHorizonStep;
                  return (
                    <button
                      key={stepIdx}
                      onClick={() => setSelectedHorizonStep(stepIdx)}
                      className={`rounded-lg border p-2 text-center font-mono text-xs transition-all cursor-pointer ${
                        isCurrent
                          ? "border-primary bg-primary/15 text-primary font-bold shadow-sm shadow-primary/20 ring-1 ring-primary"
                          : "border-border/60 bg-secondary/40 text-muted-foreground hover:bg-secondary hover:text-foreground"
                      }`}
                    >
                      <div className="text-[10px] opacity-70">
                        {stepIdx === 0 ? "OBSERVED" : `T+${stepIdx}`}
                      </div>
                      <div className="text-xs font-bold mt-0.5">
                        {stepIdx === 0 ? "NOW" : `Step ${stepIdx}`}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* NODE INSPECTOR HUD DRAWER */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
          <div className="border-b border-border/70 pb-3">
            <span className="eyebrow text-primary">ASSET TELEMETRY INSPECTOR</span>
            <h3 className="text-base font-bold text-foreground mt-0.5">
              {activeInspected.label}
            </h3>
            <span className="font-mono text-xs text-muted-foreground">
              {activeInspected.ip} · {activeInspected.sublabel}
            </span>
          </div>

          {/* Risk Level Badge */}
          <div className="rounded-lg border border-border/60 bg-secondary/40 p-3 flex items-center justify-between">
            <div>
              <span className="text-[10px] font-mono text-muted-foreground uppercase block">
                COMPUTED ASSET RISK
              </span>
              <div className="text-2xl font-mono font-bold text-foreground mt-0.5">
                {activeInspected.computedRisk}%
              </div>
            </div>

            <span
              className={`rounded px-2 py-1 font-mono text-xs font-bold uppercase ${
                activeInspected.computedRisk >= 75
                  ? "bg-destructive/15 text-destructive border border-destructive/30"
                  : activeInspected.computedRisk >= 50
                  ? "bg-warning/15 text-warning border border-warning/30"
                  : "bg-emerald-500/15 text-emerald-500 border border-emerald-500/30"
              }`}
            >
              {activeInspected.dynamicStatus}
            </span>
          </div>

          {/* Node Specifications */}
          <div className="space-y-2 text-xs font-mono">
            <div className="detail-row">
              <span>Predicted Attack Phase:</span>
              <b className="text-primary">{activeInspected.stage}</b>
            </div>
            <div className="detail-row">
              <span>Target Ports / Protocol:</span>
              <b>{activeInspected.ports}</b>
            </div>
            <div className="detail-row">
              <span>Active Concurrent Links:</span>
              <b>{activeInspected.connections} Sockets</b>
            </div>
            <div className="detail-row">
              <span>Traffic Transmission Rate:</span>
              <b>{activeInspected.trafficVolume}</b>
            </div>
          </div>

          <p className="text-xs text-muted-foreground leading-relaxed pt-1">
            {activeInspected.description}
          </p>

          {/* Quick Actions */}
          <div className="pt-2 border-t border-border/50 grid grid-cols-2 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate({ to: "/explainability" })}
              className="font-mono text-xs"
            >
              <Search size={13} /> Explain
            </Button>

            <Button
              size="sm"
              onClick={() => navigate({ to: "/what-if" })}
              className="font-mono text-xs"
            >
              <Sliders size={13} /> Simulate
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
