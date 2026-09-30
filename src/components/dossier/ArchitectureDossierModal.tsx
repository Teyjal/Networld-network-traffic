import { useState, useEffect } from "react";
import {
  X,
  FileText,
  Printer,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  ExternalLink,
  Shield,
  Layers,
  ArrowRight,
  RefreshCw,
  Cpu,
  Database,
  Search,
  Activity,
  GitBranch,
  Sliders,
  CheckCircle2,
  Terminal,
} from "lucide-react";

interface ArchitectureDossierModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ArchitectureDossierModal({
  isOpen,
  onClose,
}: ArchitectureDossierModalProps) {
  const [activePage, setActivePage] = useState<1 | 2 | "both">(1);

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-2 sm:p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-5xl rounded-2xl bg-[#061426] border border-[#0B1F3A] text-slate-100 shadow-2xl flex flex-col max-h-[96vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Dossier Control Bar */}
        <div className="flex flex-wrap items-center justify-between border-b border-[#0B1F3A] bg-[#0B1F3A]/90 px-4 py-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded bg-cyan-500/20 text-cyan-400">
              <FileText size={14} />
            </div>
            <div>
              <span className="font-bold text-white tracking-wider">
                CLASSIFIED ARCHITECTURE DOSSIER
              </span>
              <span className="hidden sm:inline text-slate-400 text-[10px] ml-2">
                DOC NW-ARCH · SMART INDIA HACKATHON
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-2 sm:mt-0">
            {/* Page Selector Tabs */}
            <div className="flex rounded-md bg-[#061426] border border-[#12345A] p-0.5">
              <button
                type="button"
                onClick={() => setActivePage(1)}
                className={`px-3 py-1 rounded text-[11px] font-bold transition-all ${
                  activePage === 1
                    ? "bg-cyan-500 text-[#061426] shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                PAGE 1: PIPELINE
              </button>
              <button
                type="button"
                onClick={() => setActivePage(2)}
                className={`px-3 py-1 rounded text-[11px] font-bold transition-all ${
                  activePage === 2
                    ? "bg-cyan-500 text-[#061426] shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                PAGE 2: MODEL & WHAT-IF
              </button>
              <button
                type="button"
                onClick={() => setActivePage("both")}
                className={`hidden md:block px-3 py-1 rounded text-[11px] font-bold transition-all ${
                  activePage === "both"
                    ? "bg-cyan-500 text-[#061426] shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                VIEW BOTH PAGES
              </button>
            </div>

            <button
              type="button"
              onClick={() => window.print()}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#12345A] text-slate-300 hover:text-white hover:bg-[#1e4675] transition-colors"
              title="Print Dossier"
            >
              <Printer size={13} />
              <span className="hidden md:inline text-[11px]">Print</span>
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded bg-red-500/20 text-red-300 hover:bg-red-500/30 hover:text-white transition-colors"
              title="Close Dossier (Esc)"
            >
              <X size={17} />
            </button>
          </div>
        </div>

        {/* Scrollable Dossier Content Area */}
        <div className="flex-1 overflow-y-auto p-3 sm:p-6 bg-[#040c17] space-y-8 print:p-0">
          {/* ========================================================================= */}
          {/* PAGE 1: SYSTEM CONCEPT, END-TO-END ARCHITECTURE & DATA PIPELINE           */}
          {/* ========================================================================= */}
          {(activePage === 1 || activePage === "both") && (
            <div className="mx-auto max-w-4xl rounded-xl border border-[#d3cbb8] bg-[#FAF7F0] text-[#1c2430] p-6 sm:p-9 shadow-2xl font-sans print:shadow-none print:border-none">
              {/* Header Stamp */}
              <div className="flex items-center justify-between border-b border-[#a89f8a]/60 pb-3 text-[10px] font-mono tracking-widest text-[#5c5446] uppercase">
                <span>CLASSIFIED RESEARCH DOSSIER · ARCHITECTURE DOCUMENT · SMART INDIA HACKATHON</span>
                <span className="font-bold text-[#1c2430]">DOC NW-ARCH / PAGE 1 OF 2</span>
              </div>

              {/* Title & Tagline */}
              <div className="mt-5 text-center">
                <h1 className="font-display text-4xl sm:text-5xl font-black tracking-tight text-[#0a1829]">
                  NETWORLD
                </h1>
                <p className="mt-1 text-base sm:text-lg font-serif italic text-[#3a4454]">
                  A Predictive Digital Twin for Proactive Network Attack Forecasting
                </p>
                <p className="mt-0.5 text-xs font-mono font-medium text-[#0284c7]">
                  “Forecast the trajectory. Explain the risk. Test the response.”
                </p>
              </div>

              {/* Metadata Banner */}
              <div className="mt-4 rounded bg-[#0b1f3a] text-white py-1.5 px-3 text-center font-mono text-[10.5px] font-bold tracking-widest uppercase">
                AI/ML ◆ CYBERSECURITY ◆ TEMPORAL NETWORK MODELING ◆ DIGITAL TWIN
              </div>

              {/* 01 PROBLEM & SYSTEM CONCEPT */}
              <div className="mt-6">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>01</span>
                  <span>PROBLEM & SYSTEM CONCEPT</span>
                </div>

                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3.5 text-xs leading-relaxed">
                  <div className="rounded-lg border border-[#d3cbb8] bg-[#f2ede2] p-3.5">
                    <b className="font-mono text-[11px] uppercase tracking-wider text-[#b91c1c] block mb-1">
                      THE PROBLEM
                    </b>
                    <p className="text-[#334155]">
                      Network attacks are not isolated events; infiltration evolves through changing network behaviour over time. Conventional traffic classifiers often analyse individual flows and primarily answer whether current traffic is malicious.
                    </p>
                  </div>

                  <div className="rounded-lg border border-[#0284c7]/40 bg-[#e0f2fe]/60 p-3.5">
                    <b className="font-mono text-[11px] uppercase tracking-wider text-[#0369a1] block mb-1">
                      THE NETWORLD APPROACH
                    </b>
                    <p className="text-[#334155]">
                      NetWorld models the network as an evolving temporal system. It learns network-state transitions, forecasts future infiltration risk, explains the evidence behind the prediction, and enables counterfactual defence simulation.
                    </p>
                  </div>
                </div>

                {/* Pipeline Banner */}
                <div className="mt-3.5 rounded bg-[#0b1f3a] text-cyan-300 p-2 text-center font-mono text-xs font-bold tracking-wider">
                  OBSERVE → MODEL → FORECAST → EXPLAIN → SIMULATE → DECIDE
                </div>
              </div>

              {/* 02 END-TO-END ARCHITECTURE */}
              <div className="mt-7">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>02</span>
                  <span>END-TO-END ARCHITECTURE</span>
                </div>

                <div className="mt-4 space-y-2.5">
                  {/* Layer 1: Input */}
                  <div className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 text-right font-mono text-[10px] text-[#64748b] uppercase tracking-wider">
                      INPUT LAYER
                    </div>
                    <div className="col-span-6 rounded border border-dashed border-[#0284c7] bg-[#f0f9ff] p-2 text-center font-mono font-bold text-[#0369a1]">
                      NETWORK TRAFFIC
                      <span className="block text-[9.5px] font-normal text-[#64748b]">CSV / PCAP / TELEMETRY</span>
                    </div>
                    <div className="col-span-3 text-[10px] font-mono text-[#64748b]">
                      Offline analysis; no external security-API dependency
                    </div>
                  </div>

                  <div className="text-center font-mono text-[#0284c7] text-xs">↓</div>

                  {/* Layer 2: Data Layer */}
                  <div className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 text-right font-mono text-[10px] text-[#64748b] uppercase tracking-wider">
                      DATA LAYER
                    </div>
                    <div className="col-span-6 rounded border border-[#cbd5e1] bg-white p-2.5 shadow-sm flex items-center gap-3">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#0b1f3a] text-white font-mono text-[10px] font-bold">1</span>
                      <div>
                        <b className="font-mono text-xs text-[#0a1829] block">TRAFFIC INGESTION</b>
                        <span className="text-[10px] text-[#64748b]">Flow + Packet Features · Timestamp + Labels</span>
                      </div>
                    </div>
                    <div className="col-span-3 text-[10px] font-mono text-[#64748b]">
                      Timestamped flows + packet-derived indicators
                    </div>
                  </div>

                  <div className="text-center font-mono text-[#0284c7] text-xs">↓</div>

                  {/* Layer 3: State Layer */}
                  <div className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 text-right font-mono text-[10px] text-[#64748b] uppercase tracking-wider">
                      STATE LAYER
                    </div>
                    <div className="col-span-6 rounded border border-[#cbd5e1] bg-white p-2.5 shadow-sm flex items-center gap-3">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#0b1f3a] text-white font-mono text-[10px] font-bold">2</span>
                      <div>
                        <b className="font-mono text-xs text-[#0a1829] block">NETWORK STATE BUILDER</b>
                        <span className="text-[10px] text-[#64748b]">Clean → Normalize → Sort · Sliding Temporal Windows</span>
                      </div>
                    </div>
                    <div className="col-span-3 text-[10px] font-mono text-[#64748b]">
                      Rolling window: S<sub>t-19</sub> → … → S<sub>t</sub>
                    </div>
                  </div>

                  <div className="text-center font-mono text-[#0284c7] text-xs">↓</div>

                  {/* Layer 4: Model Layer */}
                  <div className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 text-right font-mono text-[10px] text-[#64748b] uppercase tracking-wider">
                      MODEL LAYER
                    </div>
                    <div className="col-span-6 rounded border-2 border-[#0284c7] bg-[#f0f9ff] p-2.5 shadow-sm flex items-center gap-3">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#0284c7] text-white font-mono text-[10px] font-bold">3</span>
                      <div>
                        <b className="font-mono text-xs text-[#0369a1] block">TEMPORAL WORLD MODEL · LSTM</b>
                        <span className="text-[10px] text-[#64748b]">Learns P(S<sub>t+1</sub> | S<sub>t</sub>)</span>
                      </div>
                    </div>
                    <div className="col-span-3 text-[10px] font-mono text-[#64748b]">
                      Hidden state, next-state and risk outputs
                    </div>
                  </div>

                  <div className="text-center font-mono text-[#0284c7] text-xs">↓</div>

                  {/* Layer 5: Forecast Layer */}
                  <div className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 text-right font-mono text-[10px] text-[#64748b] uppercase tracking-wider">
                      FORECAST LAYER
                    </div>
                    <div className="col-span-6 rounded border border-[#cbd5e1] bg-white p-2.5 shadow-sm flex items-center gap-3">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#0b1f3a] text-white font-mono text-[10px] font-bold">4</span>
                      <div>
                        <b className="font-mono text-xs text-[#0a1829] block">K-STEP FORECASTING</b>
                        <span className="text-[10px] text-[#64748b]">S<sub>t</sub> → S<sub>t+1</sub> → S<sub>t+2</sub> → … · Future Infiltration Risk</span>
                      </div>
                    </div>
                    <div className="col-span-3 text-[10px] font-mono text-[#64748b]">
                      Risk estimated at every predicted step
                    </div>
                  </div>

                  {/* Intelligence Triad: SHAP / MITRE / WHAT-IF */}
                  <div className="grid grid-cols-3 gap-3 pt-3">
                    <div className="rounded border border-[#7c3aed]/40 bg-[#f5f3ff] p-2.5 text-center">
                      <b className="font-mono text-xs text-[#6d28d9] block">SHAP Explain</b>
                      <span className="text-[10px] text-[#64748b] block mt-0.5">Feature importance · Explainable predictions</span>
                    </div>

                    <div className="rounded border border-[#059669]/40 bg-[#ecfdf5] p-2.5 text-center">
                      <b className="font-mono text-xs text-[#047857] block">MITRE Mapping</b>
                      <span className="text-[10px] text-[#64748b] block mt-0.5">Attack-stage identification · Tactics & techniques</span>
                    </div>

                    <div className="rounded border border-[#b45309]/40 bg-[#fffbeb] p-2.5 text-center">
                      <b className="font-mono text-xs text-[#b45309] block">WHAT-IF Simulation</b>
                      <span className="text-[10px] text-[#64748b] block mt-0.5">Test defence actions · Compare future risk</span>
                    </div>
                  </div>

                  {/* Target Dashboard */}
                  <div className="rounded-lg bg-[#0b1f3a] text-white p-3 text-center mt-3 shadow-md">
                    <b className="font-mono text-sm tracking-wider text-cyan-300 block">
                      DEFENDER DASHBOARD
                    </b>
                    <span className="font-mono text-[10.5px] text-slate-300 tracking-wider">
                      Risk • Forecast • Stage • Evidence • Simulation
                    </span>
                  </div>
                </div>
              </div>

              {/* 03 DATA & NETWORK STATE */}
              <div className="mt-7">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>03</span>
                  <span>DATA & NETWORK STATE</span>
                </div>

                <div className="mt-3.5 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="rounded border border-[#d3cbb8] bg-[#f9f7f1] p-3">
                    <b className="font-mono text-[11px] uppercase tracking-wider text-[#0a1829] block mb-1">
                      FLOW FEATURES (36)
                    </b>
                    <p className="text-[10.5px] text-[#475569] leading-relaxed">
                      Destination Port, Protocol, Flow Duration, Forward/Backward Packets, Forward/Backward Bytes, Packet-Length Statistics, Flow Bytes/s, Flow Packets/s, Flow IAT Statistics, TCP Flags, Initial TCP Window, Active/Idle Statistics.
                    </p>
                  </div>

                  <div className="rounded border border-[#d3cbb8] bg-[#f9f7f1] p-3">
                    <b className="font-mono text-[11px] uppercase tracking-wider text-[#0a1829] block mb-1">
                      PACKET FEATURES (15)
                    </b>
                    <p className="text-[10.5px] text-[#475569] leading-relaxed">
                      TTL, TCP Window Information, Payload Size, Fragmentation Indicators, Retransmission Behaviour, Port-Scan Signatures.
                    </p>
                  </div>
                </div>

                {/* Pipeline Flow */}
                <div className="mt-3 grid grid-cols-3 sm:grid-cols-6 gap-1.5 text-center font-mono text-[9px] font-bold uppercase">
                  <div className="rounded border border-[#cbd5e1] bg-white p-2">RAW TRAFFIC</div>
                  <div className="rounded border border-[#cbd5e1] bg-white p-2">CLEANING</div>
                  <div className="rounded border border-[#cbd5e1] bg-white p-2">NORMALIZATION</div>
                  <div className="rounded border border-[#cbd5e1] bg-white p-2">CHRONO ORDER</div>
                  <div className="rounded border border-[#cbd5e1] bg-white p-2">TEMPORAL WINDOWS</div>
                  <div className="rounded border border-[#0284c7] bg-[#e0f2fe] text-[#0369a1] p-2">NETWORK STATE S<sub>t</sub></div>
                </div>

                {/* Formula Box */}
                <div className="mt-3.5 rounded-lg border border-[#0b1f3a]/40 bg-[#f1f5f9] p-3 text-center font-mono text-xs font-semibold text-[#0a1829]">
                  S<sub>t</sub> = [ traffic behaviour + packet behaviour + temporal statistics ]
                </div>

                <div className="mt-4 flex items-center justify-between border-t border-[#d3cbb8] pt-2 text-[10px] font-mono text-[#64748b]">
                  <span>Sequence of network states S<sub>t-19</sub> … S<sub>t</sub> is supplied to the temporal model (Section 04, Page 2).</span>
                  <span className="font-bold text-[#b45309]">CONTINUED ON PAGE 2 →</span>
                </div>
              </div>
            </div>
          )}

          {/* ========================================================================= */}
          {/* PAGE 2: TEMPORAL WORLD MODEL, K-STEP FORECASTING, WHAT-IF & VALIDATION    */}
          {/* ========================================================================= */}
          {(activePage === 2 || activePage === "both") && (
            <div className="mx-auto max-w-4xl rounded-xl border border-[#d3cbb8] bg-[#FAF7F0] text-[#1c2430] p-6 sm:p-9 shadow-2xl font-sans print:shadow-none print:border-none">
              {/* Header Stamp */}
              <div className="flex items-center justify-between border-b border-[#a89f8a]/60 pb-3 text-[10px] font-mono tracking-widest text-[#5c5446] uppercase">
                <span>NETWORLD · ARCHITECTURE DOSSIER</span>
                <span className="font-bold text-[#1c2430]">DOC NW-ARCH / PAGE 2 OF 2</span>
              </div>

              {/* 04 TEMPORAL WORLD MODEL */}
              <div className="mt-5">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>04</span>
                  <span>TEMPORAL WORLD MODEL</span>
                </div>

                <div className="mt-3 flex flex-wrap md:flex-nowrap items-center justify-between gap-3 rounded-lg border border-[#cbd5e1] bg-white p-4 shadow-sm">
                  <div className="flex items-center gap-1.5 font-mono text-xs">
                    <span className="rounded bg-slate-100 border border-slate-300 px-2 py-1 font-bold">S<sub>t-19</sub></span>
                    <span className="rounded bg-slate-100 border border-slate-300 px-2 py-1 font-bold">S<sub>t-18</sub></span>
                    <span className="text-slate-400">…</span>
                    <span className="rounded bg-[#0b1f3a] text-white px-2 py-1 font-bold">S<sub>t</sub></span>
                  </div>

                  <div className="font-mono text-[#0284c7] font-bold text-sm">→</div>

                  <div className="rounded-lg border-2 border-[#0284c7] bg-[#f0f9ff] px-4 py-2.5 text-center">
                    <b className="font-mono text-xs text-[#0369a1] block">LSTM WORLD MODEL</b>
                    <span className="text-[10px] text-[#64748b] block">temporal encoder + hidden state · next-state + risk heads</span>
                  </div>

                  <div className="font-mono text-[#0284c7] font-bold text-sm">→</div>

                  <div className="rounded border border-[#059669] bg-[#ecfdf5] px-3 py-2 text-center font-mono">
                    <b className="text-xs text-[#047857] block">P(S<sub>t+1</sub> | S<sub>t</sub>)</b>
                    <span className="text-[9.5px] text-[#059669]">NEXT STATE + RISK</span>
                  </div>
                </div>

                <div className="mt-2.5 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[10.5px] text-[#475569]">
                  <div className="flex items-start gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#0284c7] mt-1 shrink-0" />
                    <span>Learns temporal dependencies in network behaviour.</span>
                  </div>
                  <div className="flex items-start gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#0284c7] mt-1 shrink-0" />
                    <span>Represents current network trajectory through hidden state.</span>
                  </div>
                  <div className="flex items-start gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#0284c7] mt-1 shrink-0" />
                    <span>Supports forward prediction instead of retrospective classification.</span>
                  </div>
                </div>
              </div>

              {/* 05 K-STEP ATTACK FORECASTING */}
              <div className="mt-6">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>05</span>
                  <span>K-STEP ATTACK FORECASTING</span>
                </div>

                <div className="mt-3 flex flex-wrap md:flex-nowrap items-center justify-between gap-3 rounded-lg border border-[#cbd5e1] bg-white p-3.5">
                  <div className="text-center font-mono">
                    <span className="text-[9px] uppercase tracking-wider text-[#64748b] block mb-1 font-bold">CURRENT</span>
                    <span className="rounded bg-[#0b1f3a] text-white px-2.5 py-1 text-xs font-bold inline-block">S<sub>t</sub></span>
                    <span className="text-[9px] text-[#64748b] block mt-1">Risk now</span>
                  </div>

                  <div className="font-mono text-cyan-600 font-bold">→</div>

                  <div className="flex-1 rounded border border-cyan-200 bg-cyan-50/60 p-2.5">
                    <span className="text-[9px] uppercase tracking-wider text-[#0369a1] block mb-1 font-bold text-center font-mono">
                      FUTURE FORECAST (AUTOREGRESSIVE PREDICTION)
                    </span>
                    <div className="flex items-center justify-center gap-2 font-mono text-xs">
                      <span className="rounded bg-white border border-cyan-300 px-2 py-0.5 text-cyan-800 font-bold">T+1</span>
                      <span className="rounded bg-white border border-cyan-300 px-2 py-0.5 text-cyan-800 font-bold">T+2</span>
                      <span className="rounded bg-white border border-cyan-300 px-2 py-0.5 text-cyan-800 font-bold">T+3</span>
                      <span className="text-cyan-600">…</span>
                      <span className="rounded bg-cyan-600 text-white px-2 py-0.5 font-bold">T+K</span>
                    </div>
                    <span className="text-[8.5px] font-mono text-[#64748b] text-center block mt-1">
                      RISK AT EACH STEP (Forward rollout across temporal horizon)
                    </span>
                  </div>

                  <div className="border-l border-[#cbd5e1] pl-3 text-[10px] font-mono text-[#475569] max-w-[200px]">
                    <b className="text-[#0a1829] block">CURRENT OBS ≠ FUTURE FORECAST</b>
                    NetWorld rolls the learned state forward to estimate trajectory before compromise.
                  </div>
                </div>
              </div>

              {/* 06 EXPLAINABLE THREAT INTELLIGENCE */}
              <div className="mt-6">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>06</span>
                  <span>EXPLAINABLE THREAT INTELLIGENCE</span>
                </div>

                <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                  <div className="rounded border border-[#7c3aed]/40 bg-[#f5f3ff] p-2.5">
                    <b className="font-mono text-xs text-[#6d28d9] block">SHAP</b>
                    <p className="text-[10.5px] text-[#475569] mt-1">
                      Identifies exact flow & packet features driving risk predictions.
                    </p>
                  </div>
                  <div className="rounded border border-[#0284c7]/40 bg-[#f0f9ff] p-2.5">
                    <b className="font-mono text-xs text-[#0369a1] block">MITRE ATT&CK</b>
                    <p className="text-[10.5px] text-[#475569] mt-1">
                      Maps predicted behaviour to relevant adversary tactics & techniques.
                    </p>
                  </div>
                  <div className="rounded border border-[#059669]/40 bg-[#ecfdf5] p-2.5">
                    <b className="font-mono text-xs text-[#047857] block">EVIDENCE</b>
                    <p className="text-[10.5px] text-[#475569] mt-1">
                      Connects forecast → driving features → attack stage → traffic evidence.
                    </p>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between rounded bg-[#0b1f3a] text-white p-2 font-mono text-[9px] uppercase font-bold text-center gap-1">
                  <span className="flex-1 bg-[#12345A] py-1 rounded">RECONNAISSANCE</span>
                  <span>→</span>
                  <span className="flex-1 bg-[#12345A] py-1 rounded text-cyan-300">INITIAL ACCESS</span>
                  <span>→</span>
                  <span className="flex-1 bg-[#12345A] py-1 rounded">LATERAL MOVEMENT</span>
                  <span>→</span>
                  <span className="flex-1 bg-[#12345A] py-1 rounded">C2 BEACON</span>
                  <span>→</span>
                  <span className="flex-1 bg-[#12345A] py-1 rounded text-red-300">EXFILTRATION</span>
                </div>
              </div>

              {/* 07 COUNTERFACTUAL WHAT-IF DEFENCE */}
              <div className="mt-6">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>07</span>
                  <span>COUNTERFACTUAL WHAT-IF DEFENCE</span>
                </div>

                <div className="mt-3 rounded-lg border border-[#cbd5e1] bg-white p-3.5 space-y-2 text-xs font-mono">
                  {/* Baseline Row */}
                  <div className="flex items-center justify-between gap-2">
                    <span className="w-24 text-[10px] font-bold text-slate-500 uppercase">BASELINE</span>
                    <div className="flex items-center gap-1.5 flex-1">
                      <span className="px-2 py-0.5 rounded bg-slate-100 border text-[10.5px]">S<sub>t</sub></span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-slate-100 border text-[10.5px]">S<sub>t+1</sub></span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-slate-100 border text-[10.5px]">S<sub>t+2</sub></span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-slate-100 border text-[10.5px]">S<sub>t+3</sub></span>
                    </div>
                    <span className="px-2 py-1 rounded bg-red-100 text-red-700 font-bold text-[10px]">BASELINE RISK</span>
                  </div>

                  <div className="text-center text-[10px] text-[#64748b] font-bold">VERSUS (compare Δ reduction)</div>

                  {/* Counterfactual Row */}
                  <div className="flex items-center justify-between gap-2">
                    <span className="w-24 text-[10px] font-bold text-[#0369a1] uppercase">COUNTERFACTUAL</span>
                    <div className="flex items-center gap-1.5 flex-1">
                      <span className="px-2 py-0.5 rounded bg-cyan-100 text-cyan-900 border border-cyan-300 text-[10.5px] font-bold">S<sub>t</sub> + DEFENCE</span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-900 border text-[10.5px]">S'<sub>t+1</sub></span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-900 border text-[10.5px]">S'<sub>t+2</sub></span>
                      <span>→</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-900 border text-[10.5px]">S'<sub>t+3</sub></span>
                    </div>
                    <span className="px-2 py-1 rounded bg-emerald-100 text-emerald-700 font-bold text-[10px]">MITIGATED RISK</span>
                  </div>

                  <div className="pt-2 border-t border-slate-200 flex flex-wrap items-center justify-between text-[9.5px]">
                    <span className="text-[#64748b]">Simulated Actions:</span>
                    <span className="font-bold text-[#0a1829]">BLOCK IP • ISOLATE HOST • CLOSE PORT • RATE LIMIT • BLOCK PROTOCOL</span>
                  </div>
                </div>
              </div>

              {/* 08 CLOSED-LOOP & 09 WHY NETWORLD IS DIFFERENT */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 08 CLOSED-LOOP WORKFLOW */}
                <div>
                  <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                    <span>08</span>
                    <span>CLOSED-LOOP WORKFLOW</span>
                  </div>

                  <div className="mt-3 rounded border border-[#cbd5e1] bg-white p-3 space-y-2 text-[10.5px] font-mono">
                    <div className="flex items-center justify-between p-1.5 bg-slate-50 rounded">
                      <span>1. FORECAST</span>
                      <span className="text-cyan-700 font-bold">Lookahead T+5</span>
                    </div>
                    <div className="flex items-center justify-between p-1.5 bg-slate-50 rounded">
                      <span>2. EXPLAIN</span>
                      <span className="text-purple-700 font-bold">SHAP Drivers</span>
                    </div>
                    <div className="flex items-center justify-between p-1.5 bg-slate-50 rounded">
                      <span>3. SIMULATE DEFENCE</span>
                      <span className="text-amber-700 font-bold">Counterfactual</span>
                    </div>
                    <div className="flex items-center justify-between p-1.5 bg-slate-50 rounded">
                      <span>4. RE-RUN FORECAST</span>
                      <span className="text-blue-700 font-bold">Closed-Loop</span>
                    </div>
                    <div className="flex items-center justify-between p-1.5 bg-slate-50 rounded">
                      <span>5. COMPARE & DECIDE</span>
                      <span className="text-emerald-700 font-bold">Verified Delta</span>
                    </div>
                  </div>
                </div>

                {/* 09 WHY NETWORLD IS DIFFERENT */}
                <div>
                  <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                    <span>09</span>
                    <span>WHY NETWORLD IS DIFFERENT</span>
                  </div>

                  <div className="mt-3 rounded border border-[#cbd5e1] bg-white overflow-hidden text-[10.5px]">
                    <table className="w-full text-left">
                      <thead className="bg-[#0b1f3a] text-white font-mono text-[9.5px]">
                        <tr>
                          <th className="p-2">TRADITIONAL APPROACH</th>
                          <th className="p-2 text-cyan-300">NETWORLD</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        <tr>
                          <td className="p-2 text-slate-500">Detects suspicious traffic</td>
                          <td className="p-2 font-semibold text-slate-800">Forecasts future risk</td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-500">Analyses individual flows</td>
                          <td className="p-2 font-semibold text-slate-800">Models temporal behaviour</td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-500">Produces current alerts</td>
                          <td className="p-2 font-semibold text-slate-800">Produces risk trajectory</td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-500">Limited future-state</td>
                          <td className="p-2 font-semibold text-slate-800">Learns state transitions</td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-500">Reactive defense</td>
                          <td className="p-2 font-semibold text-slate-800">Counterfactual simulation</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* CORE INNOVATION */}
              <div className="mt-4 rounded-lg border-2 border-[#b45309] bg-[#fffbeb] p-3 text-xs text-[#92400e] leading-relaxed">
                <b className="font-mono text-[10px] tracking-wider uppercase block text-[#b45309] mb-1">
                  CORE INNOVATION
                </b>
                “NetWorld treats the network as an evolving system rather than a collection of independent traffic records. It learns how network states change, rolls the learned state forward to forecast future risk, explains the forecast, and allows defenders to test possible interventions before applying them.”
              </div>

              {/* 10 IMPLEMENTATION STACK */}
              <div className="mt-6">
                <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-[#b45309] uppercase border-b border-[#d3cbb8] pb-1">
                  <span>10</span>
                  <span>IMPLEMENTATION STACK</span>
                </div>

                <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-center font-mono text-[9.5px]">
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">DATA</b>
                    CIC-IDS2018 / PCAP
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">ML ENGINE</b>
                    PyTorch LSTM
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">XAI</b>
                    SHAP Attributor
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">SECURITY</b>
                    MITRE ATT&CK
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">BACKEND</b>
                    FastAPI
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">FRONTEND</b>
                    React + Vite
                  </div>
                  <div className="rounded border border-slate-300 bg-white p-2">
                    <b className="text-slate-400 block text-[8px]">VISUALIZATION</b>
                    Recharts / SVG
                  </div>
                </div>
              </div>

              {/* Footer Banner */}
              <div className="mt-6 rounded bg-[#0b1f3a] text-white p-3 text-center flex items-center justify-between font-mono text-xs">
                <b className="text-cyan-300 font-bold tracking-widest">NETWORLD</b>
                <span className="text-slate-300 italic">“From detecting what happened to understanding what may happen next.”</span>
                <span className="text-[10px] text-slate-400">SIH 2026</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
