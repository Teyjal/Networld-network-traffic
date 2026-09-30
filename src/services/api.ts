import type {
  TrafficUploadResponse,
  ForecastResponse,
  ExplainResponse,
  WhatIfResponse,
  ValidationResponse,
  BackendRootInfo,
  ForecastPoint,
  ForecastTimelineItem,
  MitreMapping,
  NetworkNode,
  TrafficFlow,
  ResponseStatusInfo,
  RecommendationResponse,
  ResponsePreview,
  ApplyResponseResult,
  AuditLogEntry,
  DefenderStatusResponse,
  DefenderApplyPayload,
  DefenderApplyResult,
  DefenderAuditEvent,
  DefenderThreatContext,
} from "@/types";

/**
 * Dynamically resolves the API Base URL.
 * Automatically aligns with window.location.hostname to avoid CORS and Private Network Access mismatches
 * between localhost and 127.0.0.1.
 */
export function getApiBaseUrl(): string {
  const envUrl = import.meta.env["VITE_API_BASE_URL"] as string | undefined;
  if (typeof window !== "undefined" && window.location) {
    const curHost = window.location.hostname;
    // Align with active browser host to prevent Private Network Access and CORS rejection
    if (!envUrl || envUrl.includes("127.0.0.1") || envUrl.includes("localhost")) {
      return `${window.location.protocol}//${curHost}:8000`;
    }
    return envUrl.replace(/\/$/, "");
  }
  return (envUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
}

// Keep API_BASE_URL for backward compatibility
export const API_BASE_URL = getApiBaseUrl();

/**
 * Robust fetch wrapper that automatically routes to dynamic API base URL
 * and provides clear error reporting if the backend cannot be reached.
 */
export async function apiFetch(endpointOrUrl: string, init?: RequestInit & { timeoutMs?: number }): Promise<Response> {
  const baseUrl = getApiBaseUrl();
  const url = endpointOrUrl.startsWith("http")
    ? endpointOrUrl
    : `${baseUrl}${endpointOrUrl.startsWith("/") ? "" : "/"}${endpointOrUrl}`;
  
  const timeoutMs = (init as any)?.timeoutMs || (endpointOrUrl.includes("upload") ? 120000 : 30000);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const signal = init?.signal || controller.signal;

  try {
    const res = await fetch(url, { ...init, signal });
    clearTimeout(timeoutId);
    return res;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err?.name === "AbortError") {
      throw new Error(
        `Backend request timed out after ${timeoutMs / 1000}s. The model may still be analyzing the sequence.`
      );
    }
    if (
      err?.message === "Failed to fetch" ||
      err?.name === "TypeError"
    ) {
      throw new Error(
        `Unable to reach backend at ${baseUrl}. Ensure the FastAPI server is running on port 8000.`
      );
    }
    throw err;
  }
}

/**
 * Common response handler with robust error message extraction
 */
async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail =
          typeof errJson.detail === "string"
            ? errJson.detail
            : JSON.stringify(errJson.detail);
      }
    } catch {
      try {
        const text = await res.text();
        if (text) errorDetail = text;
      } catch {}
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

/**
 * 1. Upload network traffic CSV to backend FastAPI endpoint
 * POST /api/traffic/upload
 */
export async function uploadTraffic(file: File): Promise<TrafficUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await apiFetch("/api/traffic/upload", {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
    body: formData,
  });

  return handleResponse<TrafficUploadResponse>(res);
}

/**
 * 2. Get Traffic Ingestion Status
 * GET /api/traffic/
 */
export async function getTrafficStatus(): Promise<{ status: string; message: string }> {
  const res = await apiFetch("/api/traffic/", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<{ status: string; message: string }>(res);
}

/**
 * 3. Run PyTorch LSTM K-step Temporal Forecast
 * POST /api/forecast
 * Supports direct file upload, previously uploaded filename, or latest uploaded file
 */
export async function runForecast(
  horizon = 10,
  file?: File,
  filename?: string,
  stage?: "pre" | "post" | "latest"
): Promise<ForecastResponse> {
  let url = `/api/forecast?horizon=${encodeURIComponent(horizon)}`;
  if (filename) {
    url += `&filename=${encodeURIComponent(filename)}`;
  }
  if (stage) {
    url += `&stage=${encodeURIComponent(stage)}`;
  }

  const options: RequestInit = {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
  };

  if (file) {
    const formData = new FormData();
    formData.append("file", file);
    options.body = formData;
  }

  const res = await apiFetch(url, options);
  return handleResponse<ForecastResponse>(res);
}

/**
 * Alias for fetching forecast
 */
export async function getForecast(
  horizon = 10,
  stage?: "pre" | "post" | "latest"
): Promise<ForecastResponse> {
  return runForecast(horizon, undefined, undefined, stage);
}

/**
 * 4. Compute Feature Explainability via Integrated Gradients
 * POST /api/explain
 */
export async function getExplanation(options?: {
  top_n?: number;
  sample_index?: number;
  file?: File;
  filename?: string;
  stage?: "pre" | "post" | "latest";
}): Promise<ExplainResponse> {
  const topN = options?.top_n ?? 10;
  const sampleIdx = options?.sample_index ?? -1;

  let url = `/api/explain?top_n=${encodeURIComponent(
    topN
  )}&sample_index=${encodeURIComponent(sampleIdx)}`;

  if (options?.filename) {
    url += `&filename=${encodeURIComponent(options.filename)}`;
  }
  if (options?.stage) {
    url += `&stage=${encodeURIComponent(options.stage)}`;
  }

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
  };

  if (options?.file) {
    const formData = new FormData();
    formData.append("file", options.file);
    fetchOptions.body = formData;
  }

  const res = await apiFetch(url, fetchOptions);
  return handleResponse<ExplainResponse>(res);
}

/**
 * Map user-friendly action strings to backend action keys
 */
function normalizeWhatIfAction(action: string): string {
  const lower = action.toLowerCase();
  if (lower.includes("quarantine") || lower.includes("isolate") || lower.includes("host")) return "quarantine_host";
  if (lower.includes("close") || lower.includes("port")) return "close_port";
  if (lower.includes("rate") || lower.includes("limit") || lower.includes("restrict")) return "rate_limit";
  if (lower.includes("protocol")) return "block_protocol";
  return "block_ip";
}

/**
 * 5. Run Counterfactual What-If Simulation
 * POST /api/whatif
 */
export async function runCounterfactual(params: {
  action?: string;
  target_value?: string | number | null;
  rate_factor?: number;
  horizon?: number;
  file?: File;
  filename?: string;
}): Promise<WhatIfResponse> {
  const rawAction = params.action || "block_ip";
  const actionKey = normalizeWhatIfAction(rawAction);
  const horizon = params.horizon ?? 10;
  const rateFactor = params.rate_factor ?? 0.5;

  let url = `/api/whatif?action=${encodeURIComponent(
    actionKey
  )}&horizon=${encodeURIComponent(horizon)}&rate_factor=${encodeURIComponent(rateFactor)}`;

  if (params.target_value !== undefined && params.target_value !== null) {
    url += `&target_value=${encodeURIComponent(params.target_value)}`;
  }
  if (params.filename) {
    url += `&filename=${encodeURIComponent(params.filename)}`;
  }

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
  };

  if (params.file) {
    const formData = new FormData();
    formData.append("file", params.file);
    fetchOptions.body = formData;
  }

  const res = await apiFetch(url, fetchOptions);
  return handleResponse<WhatIfResponse>(res);
}

/**
 * 6. Get Model Validation & Benchmark Metrics
 * GET /api/validation
 */
export async function getValidationMetrics(): Promise<ValidationResponse> {
  const res = await apiFetch("/api/validation", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<ValidationResponse>(res);
}

/**
 * 7. Get Backend Root System & Model Artifact Info
 * GET /
 */
export async function getSystemInfo(): Promise<BackendRootInfo> {
  const res = await apiFetch("/", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<BackendRootInfo>(res);
}

/**
 * 8. Digital Twin Network Graph
 * Note: Real-time network topology graph streaming (/api/topology/nodes) is not implemented on the backend.
 * Returns empty array when unavailable, with zero mock data and zero manufactured node fallbacks.
 */
export async function getNetworkGraph(): Promise<NetworkNode[]> {
  try {
    const res = await apiFetch("/api/topology/nodes", {
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        return data;
      }
    }
  } catch {
    // Topology endpoint is not implemented on backend
  }
  return [];
}

/**
 * 9. Traffic Flows List
 * Fetches real parsed flow records from the active dataset on the FastAPI backend
 * GET /api/traffic/flows
 */
export async function getTraffic(
  limit = 50,
  stage: "pre" | "post" | "latest" = "latest"
): Promise<TrafficFlow[]> {
  try {
    const res = await apiFetch(
      `/api/traffic/flows?limit=${limit}&stage=${stage}`,
      {
        headers: { Accept: "application/json" },
      }
    );
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        return data;
      }
    }
  } catch (err) {
    console.error("Failed to load traffic flows from backend:", err);
  }
  return [];
}

/**
 * Helper: Convert backend timeline items to chart-friendly ForecastPoint[]
 */
export function timelineToForecastPoints(
  timeline: ForecastTimelineItem[],
  mitre?: MitreMapping
): ForecastPoint[] {
  return timeline.map((item) => ({
    step: item.step === 0 ? "Current" : `T+${item.step}`,
    risk: Number((item.risk * 100).toFixed(1)),
    stage:
      mitre?.stage ||
      (item.risk_category === "HIGH"
        ? "Lateral Movement"
        : item.risk_category === "MEDIUM"
        ? "Initial Access"
        : "Reconnaissance"),
    confidence: Math.round(
      (mitre?.confidence_score
        ? mitre.confidence_score
        : item.risk > 0.5
        ? item.risk
        : 1 - item.risk) * 100
    ),
  }));
}

/**
 * 10. Response & Verification API Endpoints
 */

export async function getResponseStatus(): Promise<ResponseStatusInfo> {
  const res = await apiFetch("/api/response/status", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<ResponseStatusInfo>(res);
}

export async function setResponseMode(
  mode: string,
  is_configured?: boolean
): Promise<ResponseStatusInfo> {
  const res = await apiFetch("/api/response/mode", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ mode, is_configured }),
  });
  return handleResponse<ResponseStatusInfo>(res);
}

export async function getRecommendedResponses(
  baseline_filename?: string
): Promise<RecommendationResponse> {
  let url = `/api/response/recommend`;
  if (baseline_filename) {
    url += `?baseline_filename=${encodeURIComponent(baseline_filename)}`;
  }
  const res = await apiFetch(url, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  return handleResponse<RecommendationResponse>(res);
}

export async function previewResponse(
  action: string,
  target: string
): Promise<ResponsePreview> {
  const url = `/api/response/preview?action=${encodeURIComponent(
    action
  )}&target=${encodeURIComponent(target)}`;
  const res = await apiFetch(url, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  return handleResponse<ResponsePreview>(res);
}

export async function applyResponse(params: {
  action: string;
  target: string;
  reason?: string;
  operator_id?: string;
  approved: boolean;
  baseline_filename?: string;
}): Promise<ApplyResponseResult> {
  const res = await apiFetch("/api/response/apply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(params),
  });
  return handleResponse<ApplyResponseResult>(res);
}

export async function rollbackResponse(
  response_id: string,
  operator_id = "operator_admin"
): Promise<{ success: boolean; message: string; response_id: string }> {
  const res = await apiFetch("/api/response/rollback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ response_id, operator_id }),
  });
  return handleResponse<{ success: boolean; message: string; response_id: string }>(res);
}

export async function getResponseHistory(): Promise<{
  status: string;
  audit_log: AuditLogEntry[];
  active_rules: any[];
}> {
  const res = await apiFetch("/api/response/history", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<{
    status: string;
    audit_log: AuditLogEntry[];
    active_rules: any[];
  }>(res);
}

/**
 * 11. Authoritative Defender Closed-Loop Session API Endpoints
 */

export async function getDefenderStatus(): Promise<DefenderStatusResponse> {
  const res = await apiFetch("/api/defender/status", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<DefenderStatusResponse>(res);
}

export async function applyDefenderMitigation(
  payload: DefenderApplyPayload
): Promise<DefenderApplyResult> {
  const res = await apiFetch("/api/defender/apply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse<DefenderApplyResult>(res);
}

export async function getDefenderAudit(): Promise<{
  session_id: string | null;
  audit_events: DefenderAuditEvent[];
  count: number;
}> {
  const res = await apiFetch("/api/defender/audit", {
    headers: { Accept: "application/json" },
  });
  return handleResponse<{
    session_id: string | null;
    audit_events: DefenderAuditEvent[];
    count: number;
  }>(res);
}

export async function resetDefenderSession(): Promise<{ success: boolean; message: string }> {
  const res = await apiFetch("/api/defender/reset", {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  return handleResponse<{ success: boolean; message: string }>(res);
}

