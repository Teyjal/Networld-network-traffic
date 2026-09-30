import { useState, useEffect, useMemo } from "react";
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  Laptop,
  Server,
  Database,
  Globe,
  Radio,
  Play,
  Pause,
  Info,
  CheckCircle2,
  AlertTriangle,
  Lock,
  ArrowRight,
  Sliders,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { HelpTooltip } from "@/components/common/HelpTooltip";

export interface NetworkActivityVisualizerProps {
  currentRisk?: number | null | undefined;
  highestRisk?: number | null | undefined;
  attackStage?: string | null | undefined;
  attackerIp?: string | null | undefined;
  targetIp?: string | null | undefined;
  targetPort?: string | number | null | undefined;
  isMitigated?: boolean | undefined;
  mitigationAction?: string | null | undefined;
  activeHorizonStep?: number | undefined;
  className?: string | undefined;
}

interface NodeData {
  id: string;
  label: string;
  sublabel: string;
  ip: string;
  type: "attacker" | "internet" | "gateway" | "client" | "server" | "database";
  x: number;
  y: number;
  icon: any;
  status: "compromised" | "targeted" | "secure" | "inspecting" | "external";
  details: string;
  port?: string;
  trafficVolume?: string;
}

export function NetworkActivityVisualizer({
  currentRisk = null,
  highestRisk = null,
  attackStage = null,
  attackerIp = null,
  targetIp = null,
  targetPort = null,
  isMitigated = false,
  mitigationAction = null,
  activeHorizonStep = 0,
  className = "",
}: NetworkActivityVisualizerProps) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [simMode, setSimMode] = useState<"auto" | "normal" | "threat">("auto");

  // Determine effective simulated threat level
  const effectiveRisk = useMemo(() => {
    if (simMode === "normal") return 12;
    if (simMode === "threat") return 85;
    if (currentRisk !== null && currentRisk !== undefined) {
      // If step > 0, scale slightly with lookahead
      const boost = Math.min(25, activeHorizonStep * 2.5);
      return Math.min(100, currentRisk + boost);
    }
    return 15; // Clean baseline default
  }, [simMode, currentRisk, activeHorizonStep]);

  const hasActiveThreat = effectiveRisk >= 40 && !isMitigated;
  const isCriticalThreat = effectiveRisk >= 70 && !isMitigated;

  // Nodes positioned across a 900x380 SVG canvas
  const nodes: NodeData[] = useMemo(() => {
    const atkIp = attackerIp || "185.220.101.4";
    const tgtIp = targetIp || "10.0.0.5";
    const port = targetPort ? String(targetPort) : "445";

    return [
      {
        id: "attacker",
        label: "Attacker Host",
        sublabel: hasActiveThreat ? "Active Threat Vector" : "External Inbound",
        ip: atkIp,
        type: "attacker",
        x: 80,
        y: 110,
        icon: Radio,
        status: hasActiveThreat ? "compromised" : "external",
        details: hasActiveThreat
          ? `Host transmitting rapid connection bursts matching ${attackStage || "adversary reconnaissance"}.`
          : "External untrusted address space outside corporate perimeter.",
        trafficVolume: hasActiveThreat ? "44.2 pkts/s" : "0.4 pkts/s",
      },
      {
        id: "internet",
        label: "Remote Cloud / SaaS",
        sublabel: "Legitimate WAN Egress",
        ip: "203.0.113.15",
        type: "internet",
        x: 80,
        y: 270,
        icon: Globe,
        status: "secure",
        details: "Authorized corporate traffic communicating with encrypted partner APIs and cloud services.",
        trafficVolume: "14.8 pkts/s",
      },
      {
        id: "gateway",
        label: "Perimeter Gateway & IPS",
        sublabel: isMitigated ? "Containment Active" : "Boundary Inspector",
        ip: "10.0.0.1",
        type: "gateway",
        x: 320,
        y: 190,
        icon: isMitigated ? ShieldCheck : Shield,
        status: "inspecting",
        details: isMitigated
          ? `Active containment rule enforced (${mitigationAction || "ACL Block"}). Dropping malicious packets.`
          : "Stateful boundary device tracking flow tuples, TCP handshake flags, and byte rates.",
        trafficVolume: isMitigated ? "Filtered (Safe)" : "58.0 pkts/s",
      },
      {
        id: "client",
        label: "Workstation (VLAN 20)",
        sublabel: "Internal Endpoint",
        ip: "192.168.1.105",
        type: "client",
        x: 580,
        y: 90,
        icon: Laptop,
        status: hasActiveThreat && attackStage?.toLowerCase().includes("lateral") ? "targeted" : "secure",
        details: "Internal workstation. Communicates normally with the web application and intranet shares.",
        trafficVolume: "8.2 pkts/s",
      },
      {
        id: "server",
        label: "App & API Server",
        sublabel: `Port ${port} / HTTP-REST`,
        ip: "10.0.0.80",
        type: "server",
        x: 580,
        y: 290,
        icon: Server,
        status: hasActiveThreat && !isMitigated ? "targeted" : "secure",
        details: `Public-facing web service responding to external and internal client requests on port ${port}.`,
        port: port,
        trafficVolume: hasActiveThreat ? "38.6 pkts/s" : "12.0 pkts/s",
      },
      {
        id: "database",
        label: "Core SQL Database",
        sublabel: "Sensitive Asset Vault",
        ip: tgtIp,
        type: "database",
        x: 820,
        y: 190,
        icon: Database,
        status: isCriticalThreat ? "targeted" : "secure",
        details: "Central database repository containing customer records and transaction logs. Highest tier asset.",
        trafficVolume: "6.4 pkts/s",
      },
    ];
  }, [attackerIp, targetIp, targetPort, hasActiveThreat, isCriticalThreat, isMitigated, mitigationAction, attackStage]);

  const selectedNode = useMemo(() => {
    return nodes.find((n) => n.id === selectedNodeId) || null;
  }, [selectedNodeId, nodes]);

  // Links connecting nodes
  const links = [
    { from: "attacker", to: "gateway", isAttack: hasActiveThreat, id: "l-atk-gw" },
    { from: "internet", to: "gateway", isAttack: false, id: "l-net-gw" },
    { from: "gateway", to: "client", isAttack: false, id: "l-gw-client" },
    { from: "gateway", to: "server", isAttack: hasActiveThreat, id: "l-gw-server" },
    { from: "server", to: "database", isAttack: isCriticalThreat, id: "l-server-db" },
    { from: "client", to: "database", isAttack: false, id: "l-client-db" },
  ];

  return (
    <div className={`rounded-xl border border-border/70 bg-card p-5 shadow-sm transition-all ${className}`}>
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Radio size={16} className={isPlaying ? "animate-pulse" : ""} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-foreground">
                Network Activity Visualization
              </h3>
              <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[9.5px] font-semibold text-primary">
                LIVE TRAFFIC SIMULATION
              </span>
              <HelpTooltip
                term="Live Traffic Simulation"
                text="A visual topological animation illustrating how packets and simulated threat flows move across network boundaries. Driven by model risk calculations, not a raw packet capture socket."
              />
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Topological simulation illustrating packet flows, inspection boundaries, and attack vectors.
            </p>
          </div>
        </div>

        {/* Visualizer Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Simulation Mode Toggle */}
          <div className="segmented">
            <button
              onClick={() => setSimMode("auto")}
              className={`text-[10px] ${simMode === "auto" ? "active" : ""}`}
              title="Automatically drive flow intensity from model risk"
            >
              Auto (Model Driven)
            </button>
            <button
              onClick={() => setSimMode("normal")}
              className={`text-[10px] ${simMode === "normal" ? "active" : ""}`}
              title="Force normal healthy traffic visualization"
            >
              Baseline
            </button>
            <button
              onClick={() => setSimMode("threat")}
              className={`text-[10px] ${simMode === "threat" ? "active" : ""}`}
              title="Simulate active adversary threat traffic"
            >
              Simulate Attack
            </button>
          </div>

          {/* Play / Pause Button */}
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted/30 hover:text-foreground transition-all"
            title={isPlaying ? "Pause Flow Animation" : "Resume Flow Animation"}
          >
            {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            <span>{isPlaying ? "Pause" : "Play"}</span>
          </button>
        </div>
      </div>

      {/* Status & Legend Chip Strip */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-xs shadow-emerald-400/50" />
            <span>Authorized Network Packets</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span
              className={`h-2 w-2 rounded-full ${
                isMitigated ? "bg-amber-400" : hasActiveThreat ? "bg-rose-500 animate-ping" : "bg-muted-foreground"
              }`}
            />
            <span>
              {isMitigated
                ? "Contained / Blocked Adversary Vector"
                : hasActiveThreat
                ? "Adversary Infiltration Pulse"
                : "External Inbound Feed"}
            </span>
          </div>

          {isMitigated && (
            <div className="flex items-center gap-1.5 font-semibold text-emerald-400">
              <CheckCircle2 size={13} />
              <span>Mitigation Active · Threat Shield Engaged</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span>SIMULATED FLOW RATE:</span>
          <span className="font-bold text-foreground">
            {hasActiveThreat ? "52.4 pkts/s (Heavy)" : "18.2 pkts/s (Nominal)"}
          </span>
        </div>
      </div>

      {/* Main SVG Visualization Canvas */}
      <div className="relative mt-3 h-[340px] w-full overflow-hidden rounded-xl border border-border/80 bg-gradient-to-b from-muted/15 via-background to-muted/20 shadow-inner">
        {/* Subtle Background Grid Pattern */}
        <div
          className="absolute inset-0 opacity-[0.12] pointer-events-none"
          style={{
            backgroundImage: "radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)",
            backgroundSize: "24px 24px",
          }}
        />

        {/* SVG Flow Lines & Packet Particles */}
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox="0 0 900 380"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Linear gradients for links */}
            <linearGradient id="normalGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.5" />
            </linearGradient>

            <linearGradient id="threatGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.7" />
            </linearGradient>

            <linearGradient id="mitigatedGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.5" />
              <stop offset="60%" stopColor="#f59e0b" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.4" />
            </linearGradient>

            {/* Glowing filter for packets */}
            <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Direction Arrow Markers */}
            <marker
              id="arrow-cyan"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#06b6d4" fillOpacity="0.7" />
            </marker>

            <marker
              id="arrow-red"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#ef4444" fillOpacity="0.8" />
            </marker>
          </defs>

          {/* Draw Connection Links */}
          {links.map((link) => {
            const src = nodes.find((n) => n.id === link.from);
            const dst = nodes.find((n) => n.id === link.to);
            if (!src || !dst) return null;

            // Compute curved path
            const dx = dst.x - src.x;
            const dy = dst.y - src.y;
            const cx1 = src.x + dx * 0.45;
            const cy1 = src.y;
            const cx2 = src.x + dx * 0.55;
            const cy2 = dst.y;
            const pathData = `M ${src.x} ${src.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${dst.x} ${dst.y}`;

            const isAttackerLink = link.from === "attacker";
            const strokeColor =
              isAttackerLink && isMitigated
                ? "url(#mitigatedGrad)"
                : link.isAttack
                ? "url(#threatGrad)"
                : "url(#normalGrad)";

            const strokeDash = link.isAttack ? "6 3" : "4 4";
            const strokeWidth = link.isAttack ? 2.5 : 1.5;

            return (
              <g key={link.id}>
                {/* Background Shadow Track */}
                <path
                  d={pathData}
                  fill="none"
                  stroke={link.isAttack ? "rgba(239,68,68,0.15)" : "rgba(16,185,129,0.1)"}
                  strokeWidth={strokeWidth + 4}
                />

                {/* Animated Primary Path */}
                <path
                  id={link.id}
                  d={pathData}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                  strokeDasharray={strokeDash}
                  className={isPlaying ? "animate-pulse" : ""}
                  markerMid={link.isAttack ? "url(#arrow-red)" : "url(#arrow-cyan)"}
                />

                {/* Flowing animated packet particle (SVG SMIL animation) */}
                {isPlaying && (
                  <>
                    <circle
                      r={link.isAttack ? 4 : 3}
                      fill={link.isAttack ? "#ef4444" : "#10b981"}
                      filter={link.isAttack ? "url(#glow-red)" : "url(#glow-cyan)"}
                    >
                      <animateMotion
                        dur={link.isAttack ? "1.6s" : "3.2s"}
                        repeatCount="indefinite"
                        path={pathData}
                        keyPoints="0;1"
                        keyTimes="0;1"
                      />
                    </circle>

                    {/* Secondary trailing particle for high risk */}
                    {(link.isAttack || link.from === "internet") && (
                      <circle
                        r={link.isAttack ? 3 : 2}
                        fill={link.isAttack ? "#f59e0b" : "#06b6d4"}
                        filter={link.isAttack ? "url(#glow-red)" : "url(#glow-cyan)"}
                        opacity="0.8"
                      >
                        <animateMotion
                          dur={link.isAttack ? "1.6s" : "3.2s"}
                          begin={link.isAttack ? "0.8s" : "1.6s"}
                          repeatCount="indefinite"
                          path={pathData}
                          keyPoints="0;1"
                          keyTimes="0;1"
                        />
                      </circle>
                    )}
                  </>
                )}

                {/* Mitigation Shield Barrier on Attacker Link when mitigated */}
                {isAttackerLink && isMitigated && (
                  <g transform={`translate(${src.x + dx * 0.5 - 12}, ${src.y + dy * 0.5 - 12})`}>
                    <circle cx="12" cy="12" r="14" fill="#0f172a" stroke="#10b981" strokeWidth="2" />
                    <foreignObject x="4" y="4" width="16" height="16">
                      <ShieldCheck size={16} className="text-emerald-400" />
                    </foreignObject>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* Interactive Node Badges Positioned over SVG */}
        {nodes.map((node) => {
          const isSelected = selectedNodeId === node.id;
          const isAttacker = node.type === "attacker";
          const isGateway = node.type === "gateway";
          const isTargeted = node.status === "targeted" || (isAttacker && hasActiveThreat);

          return (
            <div
              key={node.id}
              onClick={() => setSelectedNodeId(isSelected ? null : node.id)}
              style={{
                left: `${(node.x / 900) * 100}%`,
                top: `${(node.y / 380) * 100}%`,
              }}
              className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer z-10 transition-transform duration-200 hover:scale-105"
            >
              <div
                className={`flex flex-col items-center rounded-xl p-2.5 min-w-[125px] max-w-[145px] text-center border backdrop-blur-md shadow-sm transition-all ${
                  isSelected
                    ? "border-primary ring-2 ring-primary/40 bg-card shadow-lg"
                    : isAttacker && hasActiveThreat
                    ? "border-rose-500/60 bg-rose-500/10 shadow-rose-500/20 shadow-md"
                    : isGateway && isMitigated
                    ? "border-emerald-500/60 bg-emerald-500/10 shadow-emerald-500/20 shadow-sm"
                    : isTargeted
                    ? "border-amber-500/60 bg-amber-500/10"
                    : "border-border/70 bg-card/90 hover:border-primary/50 hover:bg-card"
                }`}
              >
                {/* Node Icon with Status Dot */}
                <div className="relative mb-1.5 flex h-8 w-8 items-center justify-center rounded-lg bg-muted/40 text-foreground">
                  <node.icon
                    size={16}
                    className={
                      isAttacker && hasActiveThreat
                        ? "text-rose-400 animate-pulse"
                        : isGateway && isMitigated
                        ? "text-emerald-400"
                        : isTargeted
                        ? "text-amber-400"
                        : "text-primary"
                    }
                  />
                  {/* Status Indicator Pip */}
                  <span
                    className={`absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full border-2 border-card ${
                      isAttacker && hasActiveThreat
                        ? "bg-rose-500 animate-ping"
                        : isGateway && isMitigated
                        ? "bg-emerald-400"
                        : isTargeted
                        ? "bg-amber-400"
                        : "bg-emerald-400"
                    }`}
                  />
                </div>

                {/* Node Name & IP */}
                <span className="text-[11px] font-bold text-foreground truncate w-full">
                  {node.label}
                </span>
                <span className="font-mono text-[9.5px] text-muted-foreground truncate w-full mt-0.5">
                  {node.ip}
                </span>

                {/* Sub-label Badge */}
                <span
                  className={`mt-1.5 rounded px-1.5 py-0.5 text-[8.5px] font-mono font-semibold uppercase tracking-wider ${
                    isAttacker && hasActiveThreat
                      ? "bg-rose-500/20 text-rose-300"
                      : isGateway && isMitigated
                      ? "bg-emerald-500/20 text-emerald-300"
                      : isTargeted
                      ? "bg-amber-500/20 text-amber-300"
                      : "bg-muted/40 text-muted-foreground"
                  }`}
                >
                  {node.sublabel}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Node Telemetry HUD Drawer */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 overflow-hidden rounded-lg border border-primary/30 bg-primary/5 p-3.5 text-xs font-sans shadow-sm"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-primary/20 pb-2">
              <div className="flex items-center gap-2">
                <selectedNode.icon size={16} className="text-primary" />
                <span className="font-bold text-foreground">
                  Node Inspector: {selectedNode.label} ({selectedNode.ip})
                </span>
                <span className="rounded bg-primary/20 px-1.5 py-0.2 font-mono text-[9.5px] text-primary">
                  {selectedNode.type.toUpperCase()}
                </span>
              </div>
              <button
                onClick={() => setSelectedNodeId(null)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                ✕ Close Inspector
              </button>
            </div>

            <div className="mt-2.5 grid gap-3 sm:grid-cols-2 md:grid-cols-4 font-mono text-[11px]">
              <div className="rounded bg-background/80 p-2 border border-border/40">
                <span className="text-[10px] text-muted-foreground block font-sans">ENDPOINT IP</span>
                <span className="font-bold text-foreground">{selectedNode.ip}</span>
              </div>
              <div className="rounded bg-background/80 p-2 border border-border/40">
                <span className="text-[10px] text-muted-foreground block font-sans">TRAFFIC INTENSITY</span>
                <span className="font-bold text-primary">{selectedNode.trafficVolume}</span>
              </div>
              <div className="rounded bg-background/80 p-2 border border-border/40">
                <span className="text-[10px] text-muted-foreground block font-sans">BOUNDARY STATUS</span>
                <span
                  className={`font-bold ${
                    selectedNode.status === "compromised" || selectedNode.status === "targeted"
                      ? "text-rose-400"
                      : "text-emerald-400"
                  }`}
                >
                  {selectedNode.status.toUpperCase()}
                </span>
              </div>
              <div className="rounded bg-background/80 p-2 border border-border/40">
                <span className="text-[10px] text-muted-foreground block font-sans">MITRE TACTIC MAP</span>
                <span className="font-bold text-foreground">
                  {attackStage || "TA0001 (Initial Inbound)"}
                </span>
              </div>
            </div>

            <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed font-sans">
              <strong className="text-foreground">Node Context: </strong>
              {selectedNode.details}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
