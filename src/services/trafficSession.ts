import { useEffect, useState } from "react";
import {
  uploadTraffic,
  runForecast,
  getExplanation,
  getSystemInfo,
  getDefenderStatus,
  applyDefenderMitigation,
  resetDefenderSession,
} from "./api";
import type {
  TrafficUploadResponse,
  ForecastResponse,
  ExplainResponse,
  BackendRootInfo,
  DefenderThreatContext,
  DefenderRecommendation,
  DefenderAuditEvent,
  DefenderApplyPayload,
  DefenderApplyResult,
} from "@/types";

export type SessionTrafficStatus =
  | "INITIALIZING"
  | "NO_TRAFFIC"
  | "TRAFFIC_LOADED"
  | "PREDICTION_AVAILABLE"
  | "USING_PERSISTED"
  | "BACKEND_ERROR";

export interface TrafficSessionData {
  status: SessionTrafficStatus;
  modelReady: boolean;
  systemInfo: BackendRootInfo | null;
  sessionId: string | null;
  activeFilename: string | null;
  originalName: string | null;
  uploadTimestamp: number | null;
  isPersisted: boolean;
  uploadResult: TrafficUploadResponse | null;
  forecastResult: ForecastResponse | null;
  explainResult: ExplainResponse | null;
  
  // Threat detection context
  threatContext: DefenderThreatContext | null;
  recommendations: DefenderRecommendation[];
  preResponseRisk: number | null;
  preResponseHighestRisk: number | null;
  preResponseCategory: string | null;
  preResponseForecast: ForecastResponse | null;
  preResponseExplain: ExplainResponse | null;

  // Closed-loop mitigation & verification state
  responseStatus: string;
  activeResponse: {
    action: string;
    target: string;
    operator_id: string;
    authorization_reason: string;
    applied_at: string;
  } | null;
  flowsBlockedCount: number;
  postResponseFlowsCount: number | null;
  postResponseRisk: number | null;
  postResponseHighestRisk: number | null;
  postResponseCategory: string | null;
  postResponseForecast: ForecastResponse | null;
  postResponseExplain: ExplainResponse | null;
  riskChangePts: number | null;
  verificationStatus: string;
  verificationMessage: string | null;
  auditEvents: DefenderAuditEvent[];

  errorMessage: string | null;
}

const STORAGE_KEY = "networld_traffic_session_v4";

type Listener = (data: TrafficSessionData) => void;

class TrafficSessionManager {
  private data: TrafficSessionData;
  private listeners: Set<Listener> = new Set();
  private initPromise: Promise<void> | null = null;

  constructor() {
    this.data = this.loadFromStorage() || this.getDefaultState();
  }

  private getDefaultState(): TrafficSessionData {
    return {
      status: "NO_TRAFFIC",
      modelReady: false,
      systemInfo: null,
      sessionId: null,
      activeFilename: null,
      originalName: null,
      uploadTimestamp: null,
      isPersisted: false,
      uploadResult: null,
      forecastResult: null,
      explainResult: null,
      threatContext: null,
      recommendations: [],
      preResponseRisk: null,
      preResponseHighestRisk: null,
      preResponseCategory: null,
      preResponseForecast: null,
      preResponseExplain: null,
      responseStatus: "NONE",
      activeResponse: null,
      flowsBlockedCount: 0,
      postResponseFlowsCount: null,
      postResponseRisk: null,
      postResponseHighestRisk: null,
      postResponseCategory: null,
      postResponseForecast: null,
      postResponseExplain: null,
      riskChangePts: null,
      verificationStatus: "NOT_STARTED",
      verificationMessage: null,
      auditEvents: [],
      errorMessage: null,
    };
  }

  private loadFromStorage(): TrafficSessionData | null {
    try {
      const raw =
        (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) ||
        (typeof sessionStorage !== "undefined" && sessionStorage.getItem(STORAGE_KEY));
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (parsed && parsed.activeFilename && parsed.forecastResult) {
        return {
          ...this.getDefaultState(),
          ...parsed,
          status: "PREDICTION_AVAILABLE",
          modelReady: true,
          isPersisted: true,
        };
      }
    } catch {}
    return null;
  }

  public subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.data);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((fn) => fn(this.data));
  }

  private persistToStorage() {
    try {
      const payload = JSON.stringify({
        sessionId: this.data.sessionId,
        activeFilename: this.data.activeFilename,
        originalName: this.data.originalName,
        uploadTimestamp: this.data.uploadTimestamp,
        uploadResult: this.data.uploadResult,
        forecastResult: this.data.forecastResult,
        explainResult: this.data.explainResult,
        threatContext: this.data.threatContext,
        recommendations: this.data.recommendations,
        preResponseRisk: this.data.preResponseRisk,
        preResponseHighestRisk: this.data.preResponseHighestRisk,
        preResponseCategory: this.data.preResponseCategory,
        preResponseForecast: this.data.preResponseForecast,
        preResponseExplain: this.data.preResponseExplain,
        responseStatus: this.data.responseStatus,
        activeResponse: this.data.activeResponse,
        flowsBlockedCount: this.data.flowsBlockedCount,
        postResponseFlowsCount: this.data.postResponseFlowsCount,
        postResponseRisk: this.data.postResponseRisk,
        postResponseHighestRisk: this.data.postResponseHighestRisk,
        postResponseCategory: this.data.postResponseCategory,
        postResponseForecast: this.data.postResponseForecast,
        postResponseExplain: this.data.postResponseExplain,
        riskChangePts: this.data.riskChangePts,
        verificationStatus: this.data.verificationStatus,
        verificationMessage: this.data.verificationMessage,
        auditEvents: this.data.auditEvents,
      });

      if (typeof localStorage !== "undefined") {
        localStorage.setItem(STORAGE_KEY, payload);
      }
      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem(STORAGE_KEY, payload);
      }
    } catch {}
  }

  /**
   * Initializes session: checks backend system info and model status.
   * Restores active traffic session if a file was previously uploaded anywhere in the app.
   */
  public async initialize(): Promise<TrafficSessionData> {
    if (this.initPromise) return this.initPromise.then(() => this.data);

    this.initPromise = (async () => {
      try {
        const sys = await getSystemInfo();
        const modelLoaded = Boolean(sys?.model_artifacts?.loaded);

        // 1. Check if backend has an authoritative active session
        let backendHasSession = false;
        try {
          const st = await getDefenderStatus();
          if (st.status === "ACTIVE_SESSION" && st.session_id && (st.uploaded_filename || st.original_filename)) {
            backendHasSession = true;
            await this.syncWithBackend();
          }
        } catch {}

        if (!backendHasSession) {
          // 2. Check if a session was stored locally from an earlier upload in this project
          const restored = this.loadFromStorage();
          if (restored && restored.activeFilename && restored.forecastResult) {
            this.data = {
              ...restored,
              modelReady: modelLoaded,
              systemInfo: sys,
              errorMessage: null,
            };
          } else {
            // 3. Clean fresh state: NO traffic uploaded anywhere yet
            this.data = {
              ...this.getDefaultState(),
              modelReady: modelLoaded,
              systemInfo: sys,
              status: "NO_TRAFFIC",
              errorMessage: null,
            };
          }
        }
      } catch (err: any) {
        this.data = {
          ...this.data,
          status: this.data.activeFilename ? this.data.status : "BACKEND_ERROR",
          modelReady: false,
          errorMessage:
            err?.message || "Unable to reach FastAPI backend.",
        };
      }
      this.notify();
    })();

    return this.initPromise.then(() => this.data);
  }

  public getData(): TrafficSessionData {
    return this.data;
  }

  /**
   * Complete end-to-end analysis workflow:
   * 1. Upload CSV to backend (/api/traffic/upload)
   * 2. Synchronize authoritative session state from backend
   * 3. Update session and notify all components
   */
  public async analyzeTrafficFile(file: File): Promise<TrafficSessionData> {
    const origName = file.name;
    try {
      // 1. Ingest & validate (creates authoritative session on backend)
      const uploadRes = await uploadTraffic(file);

      // 2. Fetch authoritative session state created by backend
      const defenderState = await getDefenderStatus();

      let fc = defenderState.pre_response_forecast || (uploadRes as any).forecast || null;
      let exp = defenderState.pre_response_explain || (uploadRes as any).explain || null;

      // Ensure forecast and explainability are computed so all pages immediately have them
      if (!fc && uploadRes.filename) {
        try {
          fc = await runForecast(10, undefined, uploadRes.filename);
        } catch {}
      }
      if (!exp && uploadRes.filename) {
        try {
          exp = await getExplanation({ top_n: 5, filename: uploadRes.filename });
        } catch {}
      }

      const timestamp = Date.now();
      this.data = {
        status: "PREDICTION_AVAILABLE",
        modelReady: true,
        systemInfo: this.data.systemInfo,
        sessionId: defenderState.session_id || (uploadRes as any).session_id || null,
        activeFilename: uploadRes.filename,
        originalName: origName,
        uploadTimestamp: timestamp,
        isPersisted: true,
        uploadResult: uploadRes,
        forecastResult: fc,
        explainResult: exp,
        threatContext: defenderState.threat_context || (uploadRes as any).threat_context || null,
        recommendations: defenderState.recommendations || [],
        preResponseRisk: defenderState.pre_response_risk ?? (uploadRes as any).pre_response_risk ?? (fc ? fc.current_risk : null),
        preResponseHighestRisk: defenderState.pre_response_highest_risk || (fc ? fc.highest_predicted_risk : null),
        preResponseCategory: defenderState.pre_response_category || null,
        preResponseForecast: defenderState.pre_response_forecast || (uploadRes as any).forecast || null,
        preResponseExplain: defenderState.pre_response_explain || (uploadRes as any).explain || null,
        responseStatus: "NONE",
        activeResponse: null,
        flowsBlockedCount: 0,
        postResponseFlowsCount: null,
        postResponseRisk: null,
        postResponseHighestRisk: null,
        postResponseCategory: null,
        postResponseForecast: null,
        postResponseExplain: null,
        riskChangePts: null,
        verificationStatus: "NOT_STARTED",
        verificationMessage: null,
        auditEvents: defenderState.audit_events || [],
        errorMessage: null,
      };

      this.persistToStorage();
      this.notify();
      return this.data;
    } catch (err: any) {
      this.data = {
        ...this.data,
        errorMessage: err?.message || "Failed to process traffic dataset.",
      };
      this.notify();
      throw err;
    }
  }

  /**
   * Applies controlled-lab response and verifies mitigation using the real trained PyTorch LSTM model.
   * Immediately updates session state and notifies all subscribed pages (Command Center, Forecast, etc.)
   */
  public async applyDefenderResponse(payload: DefenderApplyPayload): Promise<DefenderApplyResult> {
    try {
      const applyResult = await applyDefenderMitigation(payload);
      const st = await getDefenderStatus();

      this.data = {
        ...this.data,
        sessionId: st.session_id || this.data.sessionId,
        responseStatus: st.response_status,
        activeResponse: st.active_response,
        flowsBlockedCount: st.flows_blocked_count,
        postResponseFlowsCount: st.post_response_flows_count,
        postResponseRisk: st.post_response_risk,
        postResponseHighestRisk: st.post_response_highest_risk || null,
        postResponseCategory: st.post_response_category || null,
        postResponseForecast: st.post_response_forecast || null,
        postResponseExplain: st.post_response_explain || null,
        riskChangePts: st.risk_change_pts,
        verificationStatus: st.verification_status,
        verificationMessage: st.verification_message,
        auditEvents: st.audit_events || [],
        // The active displayed forecast and explainability update to post-response state
        forecastResult: st.post_response_forecast || this.data.forecastResult,
        explainResult: st.post_response_explain || this.data.explainResult,
      };

      this.persistToStorage();
      this.notify();
      return applyResult;
    } catch (err: any) {
      this.data = {
        ...this.data,
        errorMessage: err?.message || "Failed to apply controlled mitigation.",
      };
      this.notify();
      throw err;
    }
  }

  /**
   * Refreshes authoritative session state directly from backend
   */
  public async syncWithBackend(): Promise<TrafficSessionData> {
    try {
      const st = await getDefenderStatus();
      if (st.status === "ACTIVE_SESSION" && st.session_id) {
        this.data = {
          ...this.data,
          sessionId: st.session_id,
          threatContext: st.threat_context,
          recommendations: st.recommendations,
          preResponseRisk: st.pre_response_risk,
          preResponseHighestRisk: st.pre_response_highest_risk || null,
          preResponseCategory: st.pre_response_category || null,
          preResponseForecast: st.pre_response_forecast || null,
          preResponseExplain: st.pre_response_explain || null,
          responseStatus: st.response_status,
          activeResponse: st.active_response,
          flowsBlockedCount: st.flows_blocked_count,
          postResponseFlowsCount: st.post_response_flows_count,
          postResponseRisk: st.post_response_risk,
          postResponseHighestRisk: st.post_response_highest_risk || null,
          postResponseCategory: st.post_response_category || null,
          postResponseForecast: st.post_response_forecast || null,
          postResponseExplain: st.post_response_explain || null,
          riskChangePts: st.risk_change_pts,
          verificationStatus: st.verification_status,
          verificationMessage: st.verification_message,
          auditEvents: st.audit_events || [],
          forecastResult: st.post_response_forecast || st.pre_response_forecast || this.data.forecastResult,
          explainResult: st.post_response_explain || st.pre_response_explain || this.data.explainResult,
        };
        this.persistToStorage();
        this.notify();
      }
    } catch {}
    return this.data;
  }

  /**
   * Set custom results if upload was performed from another sub-page
   */
  public setUploadData(
    uploadRes: TrafficUploadResponse,
    fcRes?: ForecastResponse | null,
    expRes?: ExplainResponse | null,
    originalName?: string
  ) {
    const timestamp = Date.now();
    const orig = originalName || uploadRes.filename;
    this.data = {
      ...this.data,
      status: fcRes ? "PREDICTION_AVAILABLE" : "TRAFFIC_LOADED",
      modelReady: true,
      activeFilename: uploadRes.filename,
      originalName: orig,
      uploadTimestamp: timestamp,
      isPersisted: false,
      uploadResult: uploadRes,
      forecastResult: fcRes || null,
      explainResult: expRes || null,
      errorMessage: null,
    };

    this.persistToStorage();
    this.notify();
  }

  /**
   * Clear active session and reset to NO_TRAFFIC state
   */
  public async clearSession() {
    try {
      if (typeof localStorage !== "undefined") {
        localStorage.removeItem(STORAGE_KEY);
      }
      if (typeof sessionStorage !== "undefined") {
        sessionStorage.removeItem(STORAGE_KEY);
      }
      await resetDefenderSession();
    } catch {}

    this.data = {
      ...this.getDefaultState(),
      status: "NO_TRAFFIC",
      modelReady: this.data.modelReady,
      systemInfo: this.data.systemInfo,
      errorMessage: null,
    };
    this.notify();
  }
}

export const trafficSession = new TrafficSessionManager();

export function useTrafficSession(): TrafficSessionData {
  const [data, setData] = useState<TrafficSessionData>(trafficSession.getData());

  useEffect(() => {
    const unsub = trafficSession.subscribe(setData);
    trafficSession.initialize();
    return unsub;
  }, []);

  return data;
}
