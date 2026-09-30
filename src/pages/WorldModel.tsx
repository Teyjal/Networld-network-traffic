import { useEffect, useState } from "react";
import { ArrowDown, BrainCircuit, Boxes, GitBranch, Network, Orbit, Shield, Cpu, Activity } from "lucide-react";
import { PageTitle, Panel } from "@/components/common/Panel";
import { getForecast } from "@/services/api";
import { useTrafficSession } from "@/services/trafficSession";
import type { ForecastResponse } from "@/types";

export default function WorldModel() {
  const session = useTrafficSession();
  const [forecast, setForecast] = useState<ForecastResponse | null>(
    session.forecastResult
  );
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (session.forecastResult) {
      setForecast(session.forecastResult);
    }
  }, [session.forecastResult]);

  useEffect(() => {
    if (session.activeFilename && !session.forecastResult) {
      setLoading(true);
      getForecast(5)
        .then((res) => setForecast(res))
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [session.activeFilename]);

  const totalDim = forecast?.unified_state?.total_feature_count || (forecast?.unified_state?.packet_features_available ? 51 : 36);

  const flow = [
    {
      icon: Network,
      title: "Unified Network State S(t)",
      sub: forecast?.unified_state?.packet_features_available
        ? "20 sequential flows × 51 Unified Features (36 Flow + 15 Packet)"
        : `20 sequential flows × ${totalDim} standardized features`,
    },
    {
      icon: Boxes,
      title: "Latent Temporal Representation",
      sub: "2-Layer LSTM Encoder (128 hidden dimensions)",
    },
    {
      icon: BrainCircuit,
      title: "Multi-Head Predictive World Model",
      sub: `Simultaneous State Decoder S(t+1) [${totalDim}-D] + Risk Head P(infiltration)`,
    },
    {
      icon: Orbit,
      title: "Autoregressive Forward Rollout",
      sub: "Recursive state update S(t+k) without fabricated data",
    },
  ];

  // Derive steps from real forecast timeline if available
  const futureStates =
    forecast?.timeline && forecast.timeline.length > 1
      ? forecast.timeline.slice(1, 6).map((item) => ({
          step: item.horizon_label || `S(t+${item.step})`,
          riskText: `${(item.risk * 100).toFixed(1)}% risk`,
          label:
            forecast.mitre_mapping?.stage ||
            item.risk_category ||
            "Synthesized State",
          stateType: item.is_synthesized_state ? `Synthesized ${totalDim}-D` : "Observed",
        }))
      : [1, 2, 3, 4, 5].map((n) => ({
          step: `S(t+${n})`,
          riskText: loading ? "Computing..." : "—",
          label: loading ? "Rollout active" : "Awaiting Telemetry",
          stateType: `${totalDim} Features`,
        }));

  const isMitigated = session.postResponseRisk !== null;


  return (
    <>
      <PageTitle
        title="Temporal Network World Model"
        description="Learns temporal network-state transitions, generates future network flow states, and performs forward rollout to estimate future infiltration risk."
        badge={
          isMitigated
            ? `POST-RESPONSE STATE · ${session.verificationStatus}`
            : session.isPersisted
            ? "ANALYZED TELEMETRY"
            : forecast
            ? "AUTOREGRESSIVE PREDICTION ONLINE"
            : "WORLD MODEL ONLINE"
        }
      />

      <Panel
        className="world-panel"
        title="Predictive State Transition Engine"
        eyebrow="NETWORK TRAFFIC → FUTURE STATE → FUTURE RISK"
      >
        <div className="mx-auto max-w-4xl py-2">
          <div className="space-y-1">
            {flow.map(({ icon: Icon, title, sub }, i) => (
              <div key={title} className="relative">
                {i > 0 && (
                  <div className="flex items-center justify-center my-1.5">
                    <div className="flex flex-col items-center">
                      <div className="h-3 w-px bg-border" />
                      <ArrowDown size={14} className="text-primary my-0.5" />
                      <div className="h-3 w-px bg-border" />
                    </div>
                  </div>
                )}
                <div
                  className={`model-step ${i === 2 ? "model-step-active" : ""}`}
                >
                  <div className="model-step-icon">
                    <Icon size={22} />
                  </div>
                  <div className="model-step-content">
                    <div className="flex items-center justify-between gap-3">
                      <b>{title}</b>
                      {i === 2 && (
                        <span className="rounded bg-primary/15 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-primary border border-primary/30 shrink-0">
                          MULTI-HEAD ACTIVE
                        </span>
                      )}
                    </div>
                    <span>{sub}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex flex-col items-center my-5">
            <div className="h-3 w-px bg-border" />
            <div className="rounded-full bg-primary/10 p-1.5 border border-primary/30 my-1 text-primary">
              <ArrowDown size={14} />
            </div>
            <div className="h-3 w-px bg-border" />
            <span className="font-mono text-[10px] tracking-wider text-muted-foreground uppercase mt-1">
              AUTOREGRESSIVE TEMPORAL ROLLOUT · T+1 TO T+5
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
            {futureStates.map((s) => (
              <div className="future-state group" key={s.step}>
                <span className="future-state-badge">{s.step}</span>
                <b className="font-mono text-base font-bold text-foreground my-1">{s.riskText}</b>
                <span className="text-xs text-muted-foreground truncate w-full px-1">{s.label}</span>
                <span className="mt-2 inline-block font-mono text-[9px] uppercase tracking-wider text-primary/80 bg-primary/5 px-2 py-0.5 rounded border border-primary/15">
                  {s.stateType}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Panel>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Architecture", "Multi-Head LSTM"],
          ["Input Dimension", `20 flows × ${totalDim} features`],
          [
            "State Prediction Head",
            `Synthesizes S(t+1) in ${totalDim}-D space`,
          ],
          [
            "Infiltration Head",
            forecast?.device_used
              ? `PyTorch (${forecast.device_used.toUpperCase()})`
              : "PyTorch CPU",
          ],
        ].map(([a, b]) => (
          <div className="metric-card flex flex-col justify-between" key={a}>
            <span className="eyebrow">{a}</span>
            <b className="mt-3 block font-mono text-sm text-foreground">{b}</b>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <Panel title="Multi-Head Architecture" eyebrow="NEURAL SYSTEM TOPOLOGY">
          <div className="space-y-3 text-xs text-muted-foreground">
            <div className="rounded border border-border p-3">
              <span className="font-semibold text-foreground">1. Head 1 — State Decoder:</span>
              <p className="mt-1">
                Linear(128 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 36).
                Reconstructs the next network state S(t+1) in the exact 36-feature standardized representation.
              </p>
            </div>
            <div className="rounded border border-border p-3">
              <span className="font-semibold text-foreground">2. Head 2 — Risk Predictor:</span>
              <p className="mt-1">
                Linear(128 → 64) → ReLU → Dropout → Linear(64 → 1).
                Computes raw logit for infiltration risk P(infiltration) under cross-entropy supervision.
              </p>
            </div>
            <div className="rounded border border-border p-3">
              <span className="font-semibold text-foreground">3. Head 3 — Evidence-Based ATT&CK Mapping:</span>
              <p className="mt-1">
                Deterministic MITRE translation layer that connects predicted state features and risk trajectory to kill-chain stages without fabricating training labels.
              </p>
            </div>
          </div>
        </Panel>

        <Panel title="Autoregressive Rollout Mechanics" eyebrow="RECURSIVE TRAJECTORY GENERATION">
          <div className="space-y-3 text-xs text-muted-foreground">
            <div className="flex items-start gap-3">
              <GitBranch className="mt-0.5 shrink-0 text-primary" size={16} />
              <p>
                <b>Closed-Loop Feedback:</b> The synthesized state S(t+1) is appended to the historical sequence window while the oldest flow is dropped.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <Activity className="mt-0.5 shrink-0 text-primary" size={16} />
              <p>
                <b>Trajectory Tracking:</b> Forward rollout continues iteratively for K = 1, 3, 5, or 10 steps, exposing emerging threat trajectories before compromise occurs.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <Shield className="mt-0.5 shrink-0 text-primary" size={16} />
              <p>
                <b>Counterfactual Interventions:</b> Defenders can apply hypothetical countermeasures (e.g. block IP, close port) directly onto the state vector and re-run rollout to evaluate risk reduction.
              </p>
            </div>
          </div>
        </Panel>
      </div>
    </>
  );
}
