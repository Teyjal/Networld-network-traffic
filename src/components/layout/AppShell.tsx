import { useState, useRef, useEffect } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronDown,
  CircleGauge,
  Database,
  FlaskConical,
  Globe,
  Menu,
  Moon,
  Network,
  Radar,
  RotateCcw,
  Search,
  Shield,
  ShieldCheck,
  Sparkles,
  Sun,
  Upload,
  X,
  Check,
  Terminal,
  FileText,
} from "lucide-react";
import { ArchitectureDossierModal } from "@/components/dossier/ArchitectureDossierModal";
import { Button } from "@/components/common/Button";
import { runForecast } from "@/services/api";
import { trafficSession, useTrafficSession } from "@/services/trafficSession";
import { useTheme } from "@/context/ThemeContext";
import { useLanguage, type Language } from "@/context/LanguageContext";

const navItems = [
  { to: "/", label: "COMMAND CENTER", key: "commandCenter", icon: CircleGauge },
  { to: "/traffic", label: "TRAFFIC ANALYSIS", key: "trafficAnalysis", icon: Activity },
  { to: "/world-model", label: "WORLD MODEL", key: "worldModel", icon: BrainCircuit },
  { to: "/forecast", label: "ATTACK FORECAST", key: "attackForecast", icon: ChartNoAxesCombined },
  { to: "/digital-twin", label: "DIGITAL TWIN", key: "digitalTwin", icon: Network },
  { to: "/what-if", label: "WHAT-IF SIMULATOR", key: "whatIf", icon: FlaskConical },
  { to: "/response", label: "DEFENDER RESPONSE", key: "response", icon: ShieldCheck },
  { to: "/explainability", label: "EXPLAINABILITY", key: "explainability", icon: Search },
  { to: "/validation", label: "VALIDATION", key: "validation", icon: Shield },
] as const;

export function AppShell({ children }: { children: React.ReactNode }) {
  const session = useTrafficSession();
  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t, languages } = useLanguage();

  const [open, setOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const [dossierOpen, setDossierOpen] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const langRef = useRef<HTMLDivElement>(null);

  const path = useRouterState({ select: (s) => s.location.pathname });
  const activeNavItem = navItems.find((n) => n.to === path);
  const currentTitle = activeNavItem ? activeNavItem.label : "COMMAND CENTER";

  // Close language menu on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (langRef.current && !langRef.current.contains(event.target as Node)) {
        setLangOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  async function quickUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy("upload");
    try {
      const newSession = await trafficSession.analyzeTrafficFile(f);
      const peakRisk =
        newSession.preResponseHighestRisk !== null
          ? `${(newSession.preResponseHighestRisk * 100).toFixed(1)}%`
          : "Computed";
      setNotice(`${f.name} analyzed · Peak Risk: ${peakRisk}`);
    } catch (err: any) {
      setNotice(`Upload failed: ${err?.message || "Invalid file"}`);
    } finally {
      setBusy("");
      setTimeout(() => setNotice(""), 4000);
      e.target.value = "";
    }
  }

  async function forecastNow() {
    if (!session.activeFilename) {
      setNotice("Please upload a traffic CSV file first.");
      setTimeout(() => setNotice(""), 4000);
      return;
    }
    setBusy("forecast");
    try {
      const r = await runForecast(10, undefined, session.activeFilename);
      setNotice(
        `10-step forecast updated · Peak: ${(r.highest_predicted_risk * 100).toFixed(1)}% (${r.overall_risk_category})`
      );
    } catch (err: any) {
      setNotice(`Forecast failed: ${err?.message || "Upload CSV first"}`);
    } finally {
      setBusy("");
      setTimeout(() => setNotice(""), 4000);
    }
  }

  async function handleResetSession() {
    await trafficSession.clearSession();
    setNotice("Session cleared. Ready for new traffic upload.");
    setTimeout(() => setNotice(""), 3000);
  }

  const hasActiveSession = Boolean(session.activeFilename || session.forecastResult);
  const currentLangObj =
    languages.find((l) => l.code === language) ||
    languages[0] || {
      code: "en" as Language,
      name: "English",
      nativeName: "English",
      flag: "🇺🇸",
    };

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-200">
      {/* Mobile Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Aerospace Command Sidebar - Deep Navy Anchor */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-[#0B1F3A] bg-[#061426] text-white shadow-2xl transition-transform duration-200 lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Brand Header */}
        <div className="flex h-16 items-center justify-between border-b border-[#0B1F3A] px-5">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 text-cyan-400 shadow-[0_0_14px_rgba(6,182,212,0.3)] transition-all group-hover:border-cyan-400 group-hover:scale-105">
              <Radar size={20} className="animate-spin-slow text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <b className="font-display text-base font-bold tracking-wider text-white">
                  NETWORLD
                </b>
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#06b6d4] animate-pulse" />
              </div>
              <small className="block font-mono text-[9px] font-medium tracking-widest text-slate-400 uppercase">
                Predictive Cyber Command
              </small>
            </div>
          </Link>
          <button
            className="lg:hidden p-1 text-slate-400 hover:text-white"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        {/* Command Navigation Section */}
        <div className="px-5 pt-4 pb-2">
          <p className="font-mono text-[9.5px] font-bold tracking-[0.2em] uppercase text-slate-400/90">
            COMMAND SUITE
          </p>
        </div>

        <nav className="flex-1 space-y-2 overflow-y-auto px-3.5 py-2">
          {navItems.map(({ to, label, icon: Icon }) => {
            const isActive = to === "/" ? path === "/" : path.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                onClick={() => setOpen(false)}
                className={`flex items-center justify-between rounded-lg px-3.5 py-2.5 text-[11.5px] font-mono tracking-wider transition-all min-h-[40px] ${
                  isActive
                    ? "bg-[#12345A] text-white border-l-3 border-l-[#06B6D4] shadow-[0_0_14px_rgba(6,182,212,0.28)] font-semibold"
                    : "text-slate-400 hover:bg-[#12345A]/60 hover:text-white border-l-3 border-transparent"
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    size={17}
                    className={isActive ? "text-[#06B6D4]" : "text-slate-400"}
                  />
                  <span>{label}</span>
                </div>
                {isActive && (
                  <span className="h-1.5 w-1.5 rounded-full bg-[#06B6D4] shadow-[0_0_8px_#06B6D4]" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Research Dossier Option at bottom of left sidebar */}
        <div className="px-3.5 py-2.5 border-t border-[#0B1F3A] bg-[#07172c]">
          <button
            type="button"
            onClick={() => setDossierOpen(true)}
            className="group flex w-full items-center justify-between rounded-lg border border-cyan-500/30 bg-[#0B1F3A]/80 px-3 py-2.5 text-left transition-all hover:border-cyan-400 hover:bg-[#12345A] shadow-[0_0_12px_rgba(6,182,212,0.15)] cursor-pointer"
          >
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 group-hover:scale-105 transition-transform">
                <FileText size={16} />
              </div>
              <div>
                <span className="block font-mono text-[10.5px] font-bold text-white tracking-wide group-hover:text-cyan-300 transition-colors">
                  ARCHITECTURE REPORT
                </span>
                <span className="block font-mono text-[8.5px] text-slate-400">
                  SIH Dossier (DOC NW-ARCH)
                </span>
              </div>
            </div>
            <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[8.5px] font-bold text-cyan-300 border border-cyan-500/30">
              OPEN
            </span>
          </button>
        </div>

        {/* Live Systems Telemetry Strip */}
        <div className="border-t border-[#0B1F3A] bg-[#050e1b] px-4 py-3.5">
          <div className="flex items-center justify-between mb-2.5">
            <span className="font-mono text-[9px] font-bold tracking-[0.16em] uppercase text-slate-400">
              TELEMETRY RUNTIME
            </span>
            <span className="flex items-center gap-1 font-mono text-[9px] text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              NOMINAL
            </span>
          </div>
          <div className="space-y-1.5 text-[10px] font-mono">
            <div className="flex items-center justify-between text-slate-400">
              <span>World Model (LSTM):</span>
              <span className="text-white font-semibold">ONLINE (51-D)</span>
            </div>
            <div className="flex items-center justify-between text-slate-400">
              <span>SHAP Attributor:</span>
              <span className="text-white font-semibold">ACTIVE</span>
            </div>
            <div className="flex items-center justify-between text-slate-400">
              <span>Digital Twin Engine:</span>
              <span className="text-cyan-400 font-semibold">SYNCHRONIZED</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Command Center Layout */}
      <div className="lg:pl-64">
        {/* Sticky Aerospace Header with Strong Navy Border */}
        <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b-2 border-[#0B1F3A] bg-card/95 px-4 backdrop-blur-xl md:px-6">
          <div className="flex items-center gap-3">
            <button
              className="text-muted-foreground lg:hidden p-1 hover:text-foreground"
              onClick={() => setOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            <div className="hidden items-center gap-2 font-mono text-xs sm:flex">
              <span className="font-bold tracking-wider text-[#0B1F3A] dark:text-cyan">NETWORLD</span>
              <span className="text-muted-foreground">/</span>
              <span className="text-[#061426] dark:text-foreground font-semibold tracking-wide">{currentTitle}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Telemetry Status Badge */}
            <div className="hidden items-center gap-2 rounded-md border border-[#CBD5E1] dark:border-border bg-slate-100 dark:bg-secondary/80 px-2.5 py-1 text-xs md:flex">
              <Database size={13} className="text-[#0B1F3A] dark:text-primary" />
              <span className="font-mono text-[10px] tracking-wider text-[#0B1F3A] dark:text-muted-foreground font-semibold">
                {hasActiveSession
                  ? session.originalName || "ACTIVE TELEMETRY"
                  : "CIC-IDS2017 BASELINE"}
              </span>
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  hasActiveSession ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
            </div>

            {/* Language Selector Dropdown */}
            <div className="relative" ref={langRef}>
              <button
                type="button"
                onClick={() => setLangOpen(!langOpen)}
                className="flex h-9 items-center gap-1.5 rounded-md border border-border bg-secondary px-2.5 text-xs font-medium text-foreground hover:border-primary/50 transition-colors"
                title={t("header.selectLanguage")}
                aria-label={t("header.selectLanguage")}
              >
                <Globe size={14} className="text-muted-foreground" />
                <span className="text-[13px]">{currentLangObj.flag}</span>
                <span className="font-mono text-xs hidden sm:inline">
                  {currentLangObj.code.toUpperCase()}
                </span>
                <ChevronDown size={12} className="text-muted-foreground" />
              </button>

              {langOpen && (
                <div className="absolute right-0 mt-1.5 w-44 rounded-md border border-border bg-popover p-1.5 shadow-xl z-50">
                  <div className="px-2 py-1 text-[10px] font-mono uppercase text-muted-foreground border-b border-border/50 mb-1">
                    {t("header.selectLanguage")}
                  </div>
                  {languages.map((l) => (
                    <button
                      key={l.code}
                      onClick={() => {
                        setLanguage(l.code);
                        setLangOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded px-2.5 py-1.5 text-xs transition-colors ${
                        language === l.code
                          ? "bg-primary text-primary-foreground font-semibold"
                          : "text-foreground hover:bg-secondary"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span>{l.flag}</span>
                        <span>{l.nativeName}</span>
                      </span>
                      {language === l.code && <Check size={14} />}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Theme Toggle Button (Light/Dark) */}
            <button
              type="button"
              onClick={toggleTheme}
              className="flex h-9 w-9 items-center justify-center rounded-md border border-border bg-secondary text-foreground hover:border-primary/50 transition-colors"
              title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun size={15} className="text-amber-400 transition-transform hover:rotate-45" />
              ) : (
                <Moon size={15} className="text-primary transition-transform hover:-rotate-12" />
              )}
            </button>

            {/* Reset Button */}
            {hasActiveSession && (
              <Button
                variant="outline"
                onClick={handleResetSession}
                className="hidden text-xs text-muted-foreground hover:text-foreground sm:inline-flex"
                title="Clear current traffic session and reset"
              >
                <RotateCcw size={13} />
                <span className="hidden md:inline">Reset</span>
              </Button>
            )}

            {/* Quick Upload Button */}
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={quickUpload}
              />
              <span className="inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-secondary px-3 text-xs font-semibold hover:border-primary/50 transition-colors">
                <Upload size={14} className="text-primary" />
                <span className="hidden sm:inline">
                  {busy === "upload" ? "Processing..." : "Upload Traffic"}
                </span>
              </span>
            </label>

            {/* Forecast Button */}
            <Button onClick={forecastNow} disabled={!!busy} className="shadow-sm">
              <Sparkles size={14} />
              <span className="hidden sm:inline">
                {busy === "forecast" ? "Forecasting..." : "Run Forecast"}
              </span>
            </Button>
          </div>
        </header>

        {/* View Content */}
        <main className="mx-auto max-w-[1680px] p-4 md:p-6 lg:p-8">
          {children}
        </main>
      </div>

      {/* Global Notification Toast */}
      {notice && (
        <div className="fixed bottom-5 right-5 z-[70] rounded-md border border-primary/30 bg-popover px-4 py-3 text-xs text-foreground shadow-2xl animate-in fade-in slide-in-from-bottom-2 flex items-center gap-2">
          <Terminal size={14} className="text-primary" />
          <span>{notice}</span>
        </div>
      )}

      {/* Architecture Research Dossier Modal */}
      <ArchitectureDossierModal
        isOpen={dossierOpen}
        onClose={() => setDossierOpen(false)}
      />
    </div>
  );
}
