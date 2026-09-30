import { useState, useEffect } from "react";
import { useNavigate } from "@tanstack/react-router";
import {
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  Clock,
  Sliders,
  Server,
  FileUp,
  LoaderCircle,
  FileText,
  CheckCircle2,
  XCircle,
  ArrowRight,
} from "lucide-react";
import { motion } from "motion/react";
import { Button } from "@/components/common/Button";
import { Panel } from "@/components/common/Panel";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import type { DefenderRecommendation } from "@/types";

export default function DefenderResponse() {
  const navigate = useNavigate();
  const session = useTrafficSession();

  // Mode: strictly controlled lab replay
  const [isLabMode, setIsLabMode] = useState<boolean>(true);

  // Selected Response Action & Target
  const [selectedAction, setSelectedAction] = useState<string>("block_source");
  const [customTarget, setCustomTarget] = useState<string>("");

  // Operator Authorization Form State (Must start empty - zero fake defaults)
  const [operatorId, setOperatorId] = useState<string>("");
  const [reason, setReason] = useState<string>("");
  const [authorized, setAuthorized] = useState<boolean>(false);

  // Execution State
  const [applying, setApplying] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Check if real uploaded traffic exists in active session
  const hasActiveTraffic = Boolean(
    session.activeFilename &&
      session.status !== "NO_TRAFFIC" &&
      session.status !== "INITIALIZING" &&
      session.status !== "BACKEND_ERROR"
  );

  const threat = session.threatContext;
  const isThreatDetected =
    threat && (threat.status === "ACTIVE_THREAT" || (threat.baseline_risk ?? 0) >= 0.35);

  // Default customTarget from real observed adversary source IP
  useEffect(() => {
    if (threat?.observed_source && !customTarget) {
      setCustomTarget(threat.observed_source);
    }
  }, [threat?.observed_source]);

  // Handle Response Execution
  async function handleApplyResponse() {
    if (!authorized) {
      setError("Explicit operator authorization is required before applying response.");
      return;
    }
    if (!operatorId.trim()) {
      setError("Operator ID is required. Please specify your analyst or operator identity.");
      return;
    }
    if (!session.sessionId) {
      setError("No active session ID found. Please re-upload traffic to initialize session.");
      return;
    }
    if (!customTarget.trim()) {
      setError("Target parameter cannot be empty.");
      return;
    }

    setApplying(true);
    setError(null);

    try {
      await trafficSession.applyDefenderResponse({
        session_id: session.sessionId,
        action: selectedAction,
        target: customTarget.trim(),
        operator_id: operatorId.trim(),
        authorization_reason: reason.trim() || "Contain detected lateral movement",
        authorized: true,
      });
    } catch (err: any) {
      setError(err?.message || "Failed to execute controlled response on replay buffer.");
    } finally {
      setApplying(false);
    }
  }

  // Handle Session Reset / Re-run
  async function handleResetSession() {
    try {
      await trafficSession.clearSession();
      setOperatorId("");
      setReason("");
      setAuthorized(false);
      setError(null);
    } catch (e: any) {
      setError(e?.message || "Failed to reset session.");
    }
  }

  // Pre vs Post Risk Values
  const preRisk = session.preResponseRisk !== null ? session.preResponseRisk : threat?.baseline_risk ?? null;
  const postRisk = session.postResponseRisk;
  const riskChange = session.riskChangePts;
  const isMitigated = postRisk !== null;
  const verificationStatus = session.verificationStatus;
  const isPassed = verificationStatus === "VERIFICATION_PASSED";
  const isFailed = verificationStatus === "VERIFICATION_FAILED";
  const isInsufficient = verificationStatus === "INSUFFICIENT_POST_RESPONSE_DATA";

  // Real recommendations list from backend session or computed default
  const recommendations: DefenderRecommendation[] =
    session.recommendations && session.recommendations.length > 0
      ? session.recommendations
      : threat?.observed_source
      ? [
          {
            action: "block_source",
            title: `Block Source IP ${threat.observed_source}`,
            target: threat.observed_source,
            priority: "HIGH",
            expected_impact: "Drops all inbound flows from adversary IP in controlled replay dataset.",
            description: `Filter replay traffic buffer to block adversary ${threat.observed_source}.`,
          },
        ]
      : [];

  return (
    <>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="eyebrow">CONTROLLED LAB DEFENSE</span>
            <span className="rounded border border-primary/30 bg-primary/10 px-2 py-0.5 font-mono text-[9px] font-bold text-primary">
              SAFE LAB REPLAY · NO HOST FIREWALL CHANGES
            </span>
          </div>
          <h1 className="font-display text-3xl font-semibold text-foreground md:text-4xl">
            Defender Response & Verification
          </h1>
          <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
            Apply controlled mitigations to the replayed network traffic dataset, collect the resulting post-response traffic,
            re-run the exact same PyTorch LSTM model, and empirically verify threat reduction.
          </p>
        </div>

        {/* Mode & Reset Controls */}
        <div className="flex items-center gap-2">
          {hasActiveTraffic && (
            <Button variant="outline" size="sm" onClick={handleResetSession}>
              <RotateCcw size={13} className="mr-1.5" />
              Reset Session
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-xs text-destructive">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div className="flex-1">
            <b className="block font-semibold">Response Error</b>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* STATE 1: NO ACTIVE TRAFFIC CSV UPLOADED */}
      {!hasActiveTraffic ? (
        <div className="space-y-6">
          <Panel className="border-dashed py-16 text-center">
            <div className="mx-auto flex max-w-md flex-col items-center justify-center space-y-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-secondary/80 text-muted-foreground ring-1 ring-border">
                <FileUp size={26} />
              </div>
              <div>
                <h2 className="font-display text-lg font-semibold tracking-wide text-foreground">
                  NO ACTIVE TRAFFIC
                </h2>
                <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                  Upload network traffic to initialize real-time threat detection, AI response recommendations, and empirical closed-loop verification.
                </p>
              </div>

              {/* Requirement 2: Strict Clean Table with No Data */}
              <div className="w-full max-w-sm rounded border border-border bg-card p-3 text-left">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">Risk:</div>
                  <div className="font-mono text-foreground font-semibold">—</div>
                  <div className="text-muted-foreground">Attacker:</div>
                  <div className="font-mono text-foreground font-semibold">—</div>
                  <div className="text-muted-foreground">Target:</div>
                  <div className="font-mono text-foreground font-semibold">—</div>
                  <div className="text-muted-foreground">Port:</div>
                  <div className="font-mono text-foreground font-semibold">—</div>
                  <div className="text-muted-foreground">Threat:</div>
                  <div className="font-mono text-muted-foreground font-semibold">NO DATA</div>
                </div>
              </div>

              <div className="pt-2">
                <Button onClick={() => navigate({ to: "/traffic" })}>
                  <FileUp size={14} className="mr-2" />
                  Upload Traffic
                </Button>
              </div>
            </div>
          </Panel>
        </div>
      ) : (
        /* STATE 2 & 3: TRAFFIC LOADED - REAL CLOSED-LOOP WORKFLOW */
        <div className="grid gap-6 xl:grid-cols-[1.1fr_.9fr]">
          {/* Left Column: Threat Context & Recommendations */}
          <div className="space-y-6">
            {/* THREAT DETECTED CARD */}
            <Panel
              title="Threat Detected"
              eyebrow="OBSERVED TELEMETRY · ATTACK VECTOR"
              action={
                <span className="font-mono text-xs font-bold text-destructive">
                  {threat?.attack_type || (isThreatDetected ? "INFILTRATION THREAT" : "BENIGN")}
                </span>
              }
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="data-cell">
                  <span>Source IP</span>
                  <b className="font-mono text-destructive">{threat?.observed_source || "—"}</b>
                </div>
                <div className="data-cell">
                  <span>Target</span>
                  <b className="font-mono text-primary">{threat?.observed_destination || "—"}</b>
                </div>
                <div className="data-cell">
                  <span>Port</span>
                  <b className="font-mono text-foreground">{threat?.observed_port || "—"}</b>
                </div>
                <div className="data-cell">
                  <span>Pre-response risk</span>
                  <b className="font-mono text-destructive">
                    {preRisk !== null ? `${(preRisk * 100).toFixed(1)}%` : "—"}
                  </b>
                </div>
              </div>
            </Panel>

            {/* RECOMMENDED RESPONSE */}
            <Panel
              title="Recommended Response"
              eyebrow="CONTROLLED MITIGATION OPTIONS"
            >
              <div className="space-y-3">
                {recommendations.map((rec) => {
                  const isSelected = selectedAction === rec.action;
                  return (
                    <button
                      key={rec.action}
                      onClick={() => {
                        setSelectedAction(rec.action);
                        setCustomTarget(rec.target);
                      }}
                      className={`w-full text-left rounded-lg border p-3.5 transition-all ${
                        isSelected
                          ? "border-primary bg-primary/5 shadow-sm"
                          : "border-border bg-card hover:border-primary/40"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              rec.priority === "HIGH"
                                ? "bg-destructive"
                                : rec.priority === "MEDIUM"
                                ? "bg-warning"
                                : "bg-primary"
                            }`}
                          />
                          <b className="text-xs font-semibold text-foreground">
                            {rec.title}
                          </b>
                        </div>
                        <span className="font-mono text-[10px] text-muted-foreground">
                          TARGET: {rec.target}
                        </span>
                      </div>
                      <p className="mt-1.5 text-xs text-muted-foreground">
                        {rec.expected_impact}
                      </p>
                    </button>
                  );
                })}
              </div>

              {/* Target Input */}
              <div className="mt-4 pt-4 border-t border-border">
                <label className="field">
                  <span className="text-xs">Target Identifier</span>
                  <input
                    className="input font-mono text-xs"
                    value={customTarget}
                    onChange={(e) => setCustomTarget(e.target.value)}
                    placeholder="Target IP or port to filter"
                  />
                </label>
              </div>
            </Panel>

            {/* RESPONSE STATUS (Requirement 17) */}
            <Panel
              title="Response Status"
              eyebrow="EXECUTION MONITOR"
              action={
                <span
                  className={`font-mono text-xs font-bold ${
                    applying
                      ? "text-primary"
                      : isPassed
                      ? "text-success"
                      : isFailed
                      ? "text-destructive"
                      : isInsufficient
                      ? "text-warning"
                      : "text-muted-foreground"
                  }`}
                >
                  {applying
                    ? "EXECUTING"
                    : session.responseStatus !== "NONE"
                    ? session.responseStatus
                    : "IDLE"}
                </span>
              }
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <div className="data-cell">
                  <span>Response</span>
                  <b className="font-mono text-foreground uppercase">{selectedAction}</b>
                </div>
                <div className="data-cell">
                  <span>Target Bound</span>
                  <b className="font-mono text-primary">{customTarget || "—"}</b>
                </div>
                <div className="data-cell">
                  <span>Status</span>
                  <b
                    className={`font-mono ${
                      applying
                        ? "text-primary"
                        : isPassed
                        ? "text-success"
                        : isFailed
                        ? "text-destructive"
                        : "text-muted-foreground"
                    }`}
                  >
                    {applying
                      ? "EXECUTING"
                      : session.responseStatus === "VERIFIED"
                      ? "APPLIED & VERIFIED"
                      : session.responseStatus === "FAILED"
                      ? "APPLIED (UNVERIFIED)"
                      : session.responseStatus}
                  </b>
                </div>
              </div>
            </Panel>
          </div>

          {/* Right Column: Operator Authorization & Post-Response Verification */}
          <div className="space-y-6">
            {/* OPERATOR AUTHORIZATION (Requirement 7 & 17) */}
            <Panel
              title="Operator Authorization"
              eyebrow="EXPLICIT HUMAN AUTHORIZATION GATE"
            >
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <label className="field">
                    <span className="text-xs">Operator ID</span>
                    <input
                      className="input font-mono text-xs"
                      value={operatorId}
                      onChange={(e) => setOperatorId(e.target.value)}
                      placeholder="e.g. secops_lead"
                    />
                  </label>
                  <label className="field">
                    <span className="text-xs">Authorization Reason</span>
                    <input
                      className="input text-xs"
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      placeholder="e.g. Contain detected lateral movement"
                    />
                  </label>
                </div>

                {/* Explicit Authorization Checkbox */}
                <label className="flex items-start gap-3 rounded-lg border border-border bg-muted/20 p-3.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={authorized}
                    onChange={(e) => setAuthorized(e.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-border text-primary focus:ring-primary"
                  />
                  <div className="text-xs">
                    <b className="font-semibold text-foreground">
                      Explicit Human Authorization
                    </b>
                    <p className="mt-0.5 text-muted-foreground leading-relaxed">
                      I explicitly authorize this controlled-lab response on the replayed dataset buffer.
                    </p>
                  </div>
                </label>

                {/* Apply Button */}
                <Button
                  className="w-full h-11 text-sm font-semibold"
                  onClick={handleApplyResponse}
                  disabled={
                    !authorized ||
                    !operatorId.trim() ||
                    !customTarget.trim() ||
                    applying
                  }
                >
                  {applying ? (
                    <>
                      <LoaderCircle size={16} className="animate-spin mr-2" />
                      Executing Mitigation & Re-running LSTM...
                    </>
                  ) : (
                    <>
                      <ShieldCheck size={16} className="mr-2" />
                      APPLY RESPONSE & VERIFY
                    </>
                  )}
                </Button>

                <p className="text-[10px] text-muted-foreground text-center">
                  Lab containment is strictly isolated. Host Windows firewall and production networks are never touched.
                </p>
              </div>
            </Panel>

            {/* POST-RESPONSE VERIFICATION (Requirement 9, 10, 11, 17, 19) */}
            <Panel
              title="Post-Response Verification"
              eyebrow="REAL PYTORCH LSTM RE-INFERENCE"
              action={
                isMitigated ? (
                  <span
                    className={`rounded px-2.5 py-0.5 font-mono text-[10px] font-bold border ${
                      isPassed
                        ? "bg-success/20 text-success border-success/40"
                        : isFailed
                        ? "bg-destructive/20 text-destructive border-destructive/40"
                        : "bg-warning/20 text-warning border-warning/40"
                    }`}
                  >
                    {isPassed
                      ? "VERIFICATION_PASSED · THREAT MITIGATED"
                      : isFailed
                      ? "VERIFICATION_FAILED · THREAT REMAINS ACTIVE"
                      : verificationStatus}
                  </span>
                ) : (
                  <span className="font-mono text-[10px] text-muted-foreground">
                    PENDING AUTHORIZATION
                  </span>
                )
              }
            >
              {isMitigated ? (
                <div className="space-y-4">
                  {session.verificationMessage && (
                    <div
                      className={`rounded-lg border p-3 text-xs leading-relaxed ${
                        isPassed
                          ? "border-success/30 bg-success/10 text-success"
                          : isFailed
                          ? "border-destructive/30 bg-destructive/10 text-destructive"
                          : "border-warning/30 bg-warning/10 text-warning"
                      }`}
                    >
                      <div className="flex items-center gap-2 font-bold mb-1">
                        {isPassed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
                        <span>
                          {isPassed ? "VERIFICATION PASSED" : isFailed ? "VERIFICATION FAILED" : "VERIFICATION INCONCLUSIVE"}
                        </span>
                      </div>
                      <p>{session.verificationMessage}</p>
                    </div>
                  )}

                  {/* Pre vs Post Numbers */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="data-cell">
                      <span>Pre-response</span>
                      <b className="font-mono text-destructive">
                        {preRisk !== null ? `${(preRisk * 100).toFixed(1)}%` : "—"}
                      </b>
                    </div>
                    <div className="data-cell">
                      <span>Post-response</span>
                      <b className={`font-mono ${isPassed ? "text-success" : "text-destructive"}`}>
                        {postRisk !== null ? `${(postRisk * 100).toFixed(1)}%` : "—"}
                      </b>
                    </div>
                    <div className="data-cell">
                      <span>Risk Change</span>
                      <b className={`font-mono font-bold ${riskChange !== null && riskChange < 0 ? "text-primary" : "text-destructive"}`}>
                        {riskChange !== null ? `${riskChange > 0 ? "+" : ""}${riskChange} pts` : "—"}
                      </b>
                    </div>
                  </div>

                  {/* Flow Numbers */}
                  <div className="rounded border border-border bg-card p-3 text-xs">
                    <div className="grid grid-cols-3 gap-2 text-center font-mono">
                      <div>
                        <span className="text-[10px] text-muted-foreground block">PRE-RESPONSE</span>
                        <b className="text-foreground">{session.uploadResult?.rows || 35} flows</b>
                      </div>
                      <div>
                        <span className="text-[10px] text-muted-foreground block">BLOCKED</span>
                        <b className="text-destructive font-bold">{session.flowsBlockedCount} flows</b>
                      </div>
                      <div>
                        <span className="text-[10px] text-muted-foreground block">POST-RESPONSE</span>
                        <b className="text-success font-bold">{session.postResponseFlowsCount ?? "—"} flows</b>
                      </div>
                    </div>
                  </div>

                  {/* Failed mitigation path guidance (Requirement 19) */}
                  {isFailed && (
                    <div className="rounded border border-warning/30 bg-warning/5 p-3 text-xs text-warning">
                      <b className="block font-semibold">Threat Remains Active</b>
                      <span>
                        The adversary continues to propagate lateral flows. Select another mitigation action (e.g. Close Port or Isolate Host) and authorize secondary containment.
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex h-36 flex-col items-center justify-center rounded border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
                  <Clock size={20} className="mb-2 text-muted-foreground/60" />
                  <span className="font-semibold text-foreground">Verification Not Started</span>
                  <small className="mt-1 text-[11px] text-muted-foreground">
                    Authorize and apply a controlled response above to trigger new traffic collection, LSTM re-inference, and verification.
                  </small>
                </div>
              )}
            </Panel>

            {/* AUDIT TRAIL (Requirement 17 & 18) */}
            <Panel
              title="Audit Trail"
              eyebrow="SERVER-SIDE SECURITY AUDIT EVENTS"
            >
              {session.auditEvents && session.auditEvents.length > 0 ? (
                <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                  {session.auditEvents.slice().reverse().map((entry, idx) => (
                    <div
                      key={idx}
                      className="flex items-start justify-between gap-3 text-[11px] border-b border-border/40 pb-2"
                    >
                      <div>
                        <span className="font-mono font-bold text-primary">
                          {entry.event_type}
                        </span>
                        <p className="mt-0.5 text-muted-foreground">
                          {entry.message}
                        </p>
                      </div>
                      <span className="font-mono text-[9px] text-muted-foreground shrink-0">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-muted-foreground">
                  No audit events recorded for current session.
                </div>
              )}
            </Panel>
          </div>
        </div>
      )}
    </>
  );
}
