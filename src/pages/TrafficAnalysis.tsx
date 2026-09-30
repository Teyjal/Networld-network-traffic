import { useRef, useState, useEffect } from "react";
import { FileUp, Filter, LoaderCircle, Search, AlertCircle, CheckCircle2 } from "lucide-react";
import { uploadTraffic, getTraffic } from "@/services/api";
import { useTrafficSession, trafficSession } from "@/services/trafficSession";
import { PageTitle, Panel, RiskBadge } from "@/components/common/Panel";
import type { TrafficFlow, TrafficUploadResponse } from "@/types";

export default function TrafficAnalysis() {
  const session = useTrafficSession();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<TrafficUploadResponse | null>(
    session.uploadResult
  );
  const [query, setQuery] = useState("");
  const [flowsList, setFlowsList] = useState<TrafficFlow[]>([]);

  const isMitigated = Boolean(session.postResponseRisk !== null || session.flowsBlockedCount > 0);
  const [stage, setStage] = useState<"pre" | "post">("post");

  useEffect(() => {
    setUploadResult(session.uploadResult);
  }, [session.uploadResult]);

  useEffect(() => {
    if (session.activeFilename) {
      const activeStage = isMitigated ? stage : "latest";
      getTraffic(50, activeStage).then((data) => setFlowsList(data));
    } else {
      setFlowsList([]);
    }
  }, [session.activeFilename, stage, isMitigated]);

  async function processFile(f?: File) {
    if (!f) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await trafficSession.analyzeTrafficFile(f);
      setUploadResult(updated.uploadResult);
      if (updated.uploadResult?.flows && updated.uploadResult.flows.length > 0) {
        setFlowsList(updated.uploadResult.flows);
      } else {
        const flows = await getTraffic();
        setFlowsList(flows);
      }
    } catch (err: any) {
      setError(
        err?.message || "Failed to upload and validate network traffic dataset."
      );
    } finally {
      setBusy(false);
    }
  }

  const rows = flowsList.filter((f) =>
    Object.values(f).join(" ").toLowerCase().includes(query.toLowerCase())
  );

  return (
    <>
      <PageTitle
        title="Traffic Analysis"
        description="Ingest packet captures and inspect flow-level telemetry before state encoding."
        badge={
          isMitigated
            ? `POST-RESPONSE · ${session.verificationStatus}`
            : uploadResult
            ? "DATASET VALIDATED"
            : "INGESTION READY"
        }
      />

      {isMitigated && (
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4 rounded-lg border border-primary/30 bg-primary/10 p-3 text-xs">
          <div className="flex items-center gap-4">
            <span className="font-semibold text-foreground">
              CONTROLLED RESPONSE DATASET:
            </span>
            <span className="font-mono text-muted-foreground">
              PRE-RESPONSE: <b className="text-foreground">{session.uploadResult?.rows || 35} flows</b>
            </span>
            <span className="font-mono text-destructive">
              BLOCKED: <b>{session.flowsBlockedCount} flows</b>
            </span>
            <span className="font-mono text-success">
              POST-RESPONSE: <b>{session.postResponseFlowsCount ?? "—"} flows</b>
            </span>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground">
            RULE: {session.activeResponse?.action?.toUpperCase() || "BLOCK_SOURCE"} ({session.activeResponse?.target})
          </span>
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.1fr_.9fr]">
        <Panel title="Traffic Ingestion" eyebrow="CSV NETWORK TRAFFIC (36 FEATURES)">
          <button
            onClick={() => input.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              processFile(e.dataTransfer.files[0]);
            }}
            className="drop-zone w-full"
            disabled={busy}
          >
            <input
              ref={input}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => processFile(e.target.files?.[0])}
            />
            {busy ? (
              <LoaderCircle className="mx-auto mb-3 animate-spin text-primary" size={32} />
            ) : uploadResult ? (
              <CheckCircle2 className="mx-auto mb-3 text-success" size={32} />
            ) : (
              <FileUp className="mx-auto mb-3 text-primary" size={32} />
            )}

            <b>
              {busy
                ? "Extracting & validating 36 network flow features..."
                : uploadResult
                ? `${uploadResult.filename} validated successfully`
                : "Drop network traffic CSV here"}
            </b>

            <span className="mt-2 block text-xs text-muted-foreground">
              {uploadResult
                ? `${uploadResult.rows.toLocaleString()} flows · 36 model features verified`
                : "Supported format: CSV with 36 CIC-IDS2018 model features"}
            </span>
          </button>

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
              <AlertCircle size={16} className="shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {uploadResult && (
            <div className="mt-3 flex flex-col gap-1 text-xs text-muted-foreground">
              <div className="flex items-center justify-between">
                <span
                  className={
                    uploadResult.temporal_inference_ready !== false
                      ? "text-success font-medium"
                      : "text-amber-400 font-medium"
                  }
                >
                  {uploadResult.temporal_message ||
                    (uploadResult.temporal_inference_ready !== false
                      ? "✓ Ready for LSTM forecasting"
                      : "Temporal inference pending")}
                </span>
                <span className="font-mono text-[10px]">
                  STATUS: {uploadResult.status.toUpperCase()}
                </span>
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Processing Summary" eyebrow="CAPTURE METRICS SUMMARY">
          <div className="grid grid-cols-2 gap-3">
            {[
              [
                "Total Flows",
                uploadResult
                  ? uploadResult.rows.toLocaleString()
                  : flowsList.length > 0
                  ? `${flowsList.length} flows`
                  : "—",
              ],
              [
                "Model Features",
                uploadResult
                  ? `${uploadResult.model_features} / 36 verified`
                  : flowsList.length > 0
                  ? "36 / 36 verified"
                  : "—",
              ],
              [
                "Protocols",
                uploadResult?.protocols &&
                Object.keys(uploadResult.protocols).length > 0
                  ? Object.entries(uploadResult.protocols)
                      .map(
                        ([k, v]) =>
                          `${k === "6" ? "TCP" : k === "17" ? "UDP" : k}: ${v}`
                      )
                      .join(", ")
                  : flowsList.length > 0
                  ? Array.from(new Set(flowsList.map((f) => f.protocol))).join(", ")
                  : "—",
              ],
              [
                "Missing Values",
                uploadResult
                  ? uploadResult.missing_values.toString()
                  : flowsList.length > 0
                  ? "0"
                  : "—",
              ],
              [
                "Unique Sources",
                uploadResult?.source_info?.unique_src_ips
                  ? `${uploadResult.source_info.unique_src_ips} IPs`
                  : uploadResult?.source_info?.unique_src_ports
                  ? `${uploadResult.source_info.unique_src_ports} ports`
                  : flowsList.length > 0
                  ? `${new Set(flowsList.map((f) => f.source)).size} IPs`
                  : "—",
              ],
              [
                "Attack Labels",
                uploadResult?.label_distribution &&
                Object.keys(uploadResult.label_distribution).length > 0
                  ? Object.entries(uploadResult.label_distribution)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")
                  : flowsList.length > 0
                  ? Object.entries(
                      flowsList.reduce((acc, f) => {
                        const lbl = f.label || "Benign";
                        acc[lbl] = (acc[lbl] || 0) + 1;
                        return acc;
                      }, {} as Record<string, number>)
                    )
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")
                  : "—",
              ],
            ].map((x) => (
              <div className="data-cell" key={x[0]}>
                <span>{x[0]}</span>
                <b>{x[1]}</b>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel
        className="mt-6"
        title="Flow Inspector"
        eyebrow="TELEMETRY FLOW STREAM"
        action={
          <div className="flex items-center gap-3">
            {isMitigated && (
              <div className="segmented">
                <button
                  onClick={() => setStage("pre")}
                  className={stage === "pre" ? "active" : ""}
                >
                  PRE-RESPONSE
                </button>
                <button
                  onClick={() => setStage("post")}
                  className={stage === "post" ? "active" : ""}
                >
                  POST-RESPONSE ({session.postResponseFlowsCount ?? "—"})
                </button>
              </div>
            )}
            <label className="relative">
              <Search
                size={13}
                className="absolute left-3 top-2.5 text-muted-foreground"
              />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter traffic"
                className="input h-8 pl-8"
              />
            </label>
          </div>
        }
      >
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-12 text-center">#</th>
                <th className="text-left">Timestamp</th>
                <th className="text-left">Source IP</th>
                <th className="text-left">Destination IP</th>
                <th className="text-right">Src Port</th>
                <th className="text-right">Dst Port</th>
                <th className="text-center">Protocol</th>
                <th className="text-right">Packets</th>
                <th className="text-right">Bytes</th>
                <th className="text-right">Duration</th>
                <th className="text-left">Attack Classification</th>
                <th className="text-center">Risk Assessment</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((f, i) => (
                <tr key={i}>
                  <td className="w-12 text-center font-mono text-xs text-muted-foreground">{f.index || i + 1}</td>
                  <td className="text-left font-mono text-xs whitespace-nowrap text-muted-foreground">{f.time}</td>
                  <td className="text-left font-mono text-xs whitespace-nowrap font-medium text-foreground">{f.source}</td>
                  <td className="text-left font-mono text-xs whitespace-nowrap font-medium text-foreground">{f.destination}</td>
                  <td className="text-right font-mono text-xs text-muted-foreground">{f.sourcePort}</td>
                  <td className="text-right font-mono text-xs text-muted-foreground">{f.destinationPort}</td>
                  <td className="text-center">
                    <span className="tag">{f.protocol}</span>
                  </td>
                  <td className="text-right font-mono text-xs">{typeof f.packets === "number" ? f.packets.toLocaleString() : f.packets}</td>
                  <td className="text-right font-mono text-xs">{f.bytes}</td>
                  <td className="text-right font-mono text-xs">{f.duration}</td>
                  <td className="text-left">
                    {f.label ? (
                      <span
                        className={`tag font-mono text-[10px] uppercase font-semibold ${
                          f.label.toLowerCase().includes("infiltrat") ||
                          f.label.toLowerCase().includes("attack") ||
                          f.label.toLowerCase().includes("dos") ||
                          f.label.toLowerCase().includes("bot")
                            ? "border-destructive/40 text-destructive bg-destructive/10"
                            : "border-success/40 text-success bg-success/10"
                        }`}
                      >
                        {f.label}
                      </span>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </td>
                  <td className="text-center">
                    <div className="inline-flex justify-center">
                      <RiskBadge value={f.risk} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && !busy && (
            <div className="py-12 text-center text-sm text-muted-foreground">
              <Filter className="mx-auto mb-2" />
              {query
                ? "No flows match this filter."
                : "No flows loaded yet. Upload a network traffic CSV above to inspect live telemetry."}
            </div>
          )}
          {busy && (
            <div className="py-12 text-center text-sm text-muted-foreground">
              <LoaderCircle className="mx-auto mb-2 animate-spin text-primary" size={24} />
              Processing traffic...
            </div>
          )}
        </div>
      </Panel>
    </>
  );
}
