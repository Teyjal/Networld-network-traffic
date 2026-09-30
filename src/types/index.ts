export type RiskLevel = "low" | "medium" | "high" | "critical";

export type TrafficFlow = {
  index?: number;
  time: string;
  source: string;
  destination: string;
  sourcePort: number | string;
  destinationPort: number | string;
  protocol: string;
  packets: number;
  bytes: string;
  duration: string;
  risk: number | null;
  riskScore?: number | null;
  label?: string | null;
  isWarmup?: boolean;
};

export type ForecastPoint = {
  step: string;
  risk: number;
  stage: string;
  confidence: number;
  baseline?: number;
  counterfactual?: number;
};

export type NetworkNode = {
  id: string;
  label: string;
  type: string;
  ip: string;
  risk: number;
  connections: number;
  ports: string;
  stage: string;
  traffic: string;
  x: number;
  y: number;
  status: RiskLevel | "monitored";
};

export type ValidationMetric = {
  metric: string;
  worldModel?: number;
  lstm?: number;
  baseline: number;
};

// ==========================================
// Real Backend FastAPI API Response Schemas
// ==========================================

export interface TrafficUploadResponse {
  success: boolean;
  filename: string;
  rows: number;
  columns: number;
  missing_values: number;
  model_features: number;
  packet_features_available?: boolean;
  packet_features_count?: number;
  flow_features_count?: number;
  total_features?: number;
  packet_metrics?: Record<string, number>;
  protocols: Record<string, number>;
  source_info?: {
    unique_src_ports?: number;
    unique_src_ips?: number;
  };
  destination_info?: {
    unique_dst_ports?: number;
    unique_dst_ips?: number;
  };
  label_distribution?: Record<string, number>;
  temporal_inference_ready?: boolean;
  temporal_status?: string;
  temporal_message?: string;
  flows?: TrafficFlow[];
  status: string;
  message: string;
}

export interface ForecastTimelineItem {
  step: number;
  risk: number;
  risk_category: string;
  status: "CURRENT" | "FORECAST";
  description: string;
  horizon_label?: string;
  key_features?: Record<string, number>;
  is_synthesized_state?: boolean;
}

export interface ForecastTrajectoryStep {
  step: number;
  horizon_label: string;
  status: "CURRENT" | "FORECAST";
  risk_probability: number;
  risk_percent: number;
  risk_level: string;
  key_features: Record<string, number>;
  is_synthesized_state: boolean;
  description: string;
}

export interface AttackStageProgressionItem {
  stage: string;
  tactic: string;
  tactic_name: string;
  technique_id: string;
  technique_name: string;
  sub_technique?: string;
  supported: boolean;
  confidence: number;
  confidence_percent: number;
  rules_satisfied: number;
  supporting_evidence: string[];
  formatted_output: string;
}

export interface MitreMapping {
  stage: string;
  predicted_stage?: string;
  tactic?: string;
  technique_id?: string;
  technique_name?: string;
  mapping_type?: string;
  dataset_label?: string;
  confidence_score?: number;
  confidence?: number;
  confidence_percent?: number;
  supporting_evidence?: string[];
  formatted_output?: string;
  progression?: AttackStageProgressionItem[];
  supported_stages_count?: number;
  evidence?: {
    rules_matched?: string[];
    risk_probability?: number;
    label_source?: string;
    supported_stages?: string[];
  };
  description?: string;
}

export interface UnifiedNetworkStateInfo {
  supported: boolean;
  packet_features_available: boolean;
  flow_feature_count: number;
  packet_feature_count: number;
  total_feature_count: number;
  active_model_dim: number;
  source: string;
  packet_features?: Record<string, number>;
  description?: string;
}

export interface ForecastResponse {
  success: boolean;
  filename: string;
  horizon: number;
  total_available_windows?: number;
  current_risk: number;
  current_risk_category: string;
  highest_predicted_risk: number;
  overall_risk_category: string;
  peak_risk_horizon?: string;
  mitre_mapping: MitreMapping;
  timeline: ForecastTimelineItem[];
  trajectory?: ForecastTrajectoryStep[];
  top_risk_drivers?: FeatureAttribution[];
  state_synthesis_supported?: boolean;
  unified_state?: UnifiedNetworkStateInfo;
  model_version?: string;
  id?: string;
  device_used?: string;
  status: string;
  disclaimer?: string;
}


export interface FeatureAttribution {
  feature: string;
  importance: number;
  direction: "increases_risk" | "decreases_risk";
  raw_value: number;
  shap_value?: number;
  contribution_pct?: number;
}

export interface TimestepContribution {
  step_index: number;
  label: string;
  contribution_pct: number;
}

export interface WaterfallStep {
  step: string;
  feature: string;
  delta: number;
  cumulative: number;
  direction: string;
}

export interface ExplainResponse {
  success: boolean;
  filename: string;
  sample_index?: number;
  prediction: number;
  predicted_risk_percent?: number;
  overall_risk_category: string;
  base_value?: number;
  top_features: FeatureAttribution[];
  timestep_contributions?: number[];
  top_timesteps?: TimestepContribution[];
  waterfall?: WaterfallStep[];
  method_used: string;
  sequence_window_flows?: number;
  total_features_evaluated?: number;
  status: string;
  disclaimer?: string;
  description?: string;
}

export interface WhatIfTimelineItem extends ForecastTimelineItem {}

export interface WhatIfStepComparison {
  step: number;
  horizon_label: string;
  baseline_risk: number;
  baseline_risk_percent: number;
  baseline_risk_level: string;
  whatif_risk: number;
  whatif_risk_percent: number;
  whatif_risk_level: string;
  risk_delta: number;
  risk_delta_percent: number;
  risk_reduced: boolean;
}

export interface WhatIfRiskState {
  risk: number;
  peak_risk?: number;
  current_risk: number;
  risk_category: string;
  timeline: ForecastTimelineItem[];
  trajectory?: ForecastTrajectoryStep[];
}

export interface WhatIfResponse {
  success: boolean;
  filename: string;
  action: string;
  selected_action?: string;
  description?: string;
  target_value?: string | number | null;
  simulation: boolean;
  simulation_type: string;
  model_version?: string;
  affected_features?: string[];
  affected_records: number;
  affected_records_percentage: number;
  baseline: WhatIfRiskState;
  counterfactual: WhatIfRiskState;
  whatif?: WhatIfRiskState;
  baseline_trajectory?: ForecastTrajectoryStep[];
  counterfactual_trajectory?: ForecastTrajectoryStep[];
  whatif_trajectory?: ForecastTrajectoryStep[];
  step_comparisons?: WhatIfStepComparison[];
  risk_change: number;
  risk_change_percent?: number;
  mitigation_percentage?: number;
  risk_reduced?: boolean;
  status: string;
  summary: string;
  disclaimer: string;
}

export interface ModelEvaluationMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  confusion_matrix?: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    matrix: number[][];
  };
}

export interface ValidationResponse {
  status: string;
  dataset_info?: {
    training_dataset: string;
    evaluation_split: string;
    total_test_samples: number;
    sequence_length: number;
    feature_count: number;
    benign_samples: number;
    attack_samples: number;
  };
  models?: {
    networld_lstm: ModelEvaluationMetrics;
    logistic_regression: ModelEvaluationMetrics;
    world_model?: ModelEvaluationMetrics;
  };
  message?: string;
}

export interface BackendRootInfo {
  name: string;
  description: string;
  version: string;
  status: string;
  mode: string;
  model_artifacts?: {
    model_exists: boolean;
    model_path: string;
    scaler_exists: boolean;
    scaler_path: string;
    expected_input_shape: [number, number];
    device: string;
    loaded: boolean;
  };
  supported_input?: {
    sequence_length: number;
    num_features: number;
    dataset: string;
  };
}

export interface ResponseRecommendation {
  action: string;
  title: string;
  target: string;
  priority: "HIGH" | "MEDIUM" | "LOW";
  expected_impact: string;
  description: string;
}

export interface ThreatContext {
  observed_source: string;
  observed_destination: string;
  observed_port: string;
  attack_type: string;
  risk_score: number;
  baseline_filename?: string;
}

export interface RecommendationResponse {
  status: string;
  threat_context: ThreatContext;
  recommendations: ResponseRecommendation[];
  mode: string;
  is_configured: boolean;
}

export interface ResponsePreview {
  valid: boolean;
  action: string;
  target: string;
  lab_environment: string;
  impact_scope: string;
  description: string;
  rollback_supported: boolean;
  error?: string;
}

export interface VerificationMetrics {
  before_risk: number;
  after_risk: number;
  before_current_risk: number;
  after_current_risk: number;
  risk_delta: number;
  risk_reduction_pct: number;
  before_attack_flows: number;
  after_attack_flows: number;
  before_high_risk_flows: number;
  after_high_risk_flows: number;
  total_flows_observed: number;
}

export interface VerificationResult {
  success: boolean;
  response_id: string;
  action: string;
  target: string;
  verification_status:
    | "MITIGATED"
    | "RISK_REDUCED"
    | "PARTIALLY_MITIGATED"
    | "STILL_ACTIVE"
    | "VERIFICATION_FAILED";
  status_description: string;
  verification_timestamp: string;
  metrics: VerificationMetrics;
  post_forecast: {
    current_risk: number;
    highest_predicted_risk: number;
    mitre_mapping?: MitreMapping;
    timeline: ForecastTimelineItem[];
  };
  model_verification: {
    architecture: string;
    artifact: string;
    features_verified: number;
    sequence_window: number;
    device: string;
  };
  error?: string;
}

export interface ResponseRecord {
  response_id: string;
  timestamp: string;
  action: string;
  target: string;
  reason: string;
  requested_by: string;
  approval_status: "APPROVED" | "PENDING_APPROVAL" | "REJECTED";
  execution_status: "APPLIED" | "OBSERVING" | "ROLLED_BACK" | "FAILED";
  before_risk: number;
  post_response_risk: number;
  verification_status: string;
  verification_timestamp: string;
  rollback_available: boolean;
}

export interface ApplyResponseResult {
  success: boolean;
  response_record: ResponseRecord;
  verification: VerificationResult;
  mode: string;
}

export interface AuditLogEntry {
  timestamp: string;
  event_type: string;
  details: string;
  response_id?: string;
}

export interface ResponseStatusInfo {
  mode: "CONTROLLED_LAB" | "REPLAY";
  is_configured: boolean;
  active_rules_count: number;
  active_rules: any[];
  audit_log_count: number;
  allowed_actions: string[];
}

export interface DefenderThreatContext {
  status: "ACTIVE_THREAT" | "BENIGN";
  attack_type: string;
  observed_source: string;
  observed_destination: string;
  observed_port: string;
  baseline_risk: number;
  highest_risk: number;
  risk_category: string;
}

export interface DefenderRecommendation {
  action: string;
  title: string;
  target: string;
  priority: "HIGH" | "MEDIUM" | "LOW";
  expected_impact: string;
  description: string;
}

export interface DefenderAuditEvent {
  event_type: string;
  timestamp: string;
  session_id: string;
  message: string;
  status: string;
}

export interface DefenderStatusResponse {
  status: "ACTIVE_SESSION" | "NO_TRAFFIC";
  session_id: string | null;
  uploaded_filename?: string;
  original_filename?: string;
  threat_context: DefenderThreatContext | null;
  recommendations: DefenderRecommendation[];
  pre_response_risk: number | null;
  pre_response_highest_risk?: number | null;
  pre_response_category?: string;
  response_status: string;
  active_response: {
    action: string;
    target: string;
    operator_id: string;
    authorization_reason: string;
    applied_at: string;
  } | null;
  flows_blocked_count: number;
  post_response_flows_count: number | null;
  post_response_risk: number | null;
  post_response_highest_risk?: number | null;
  post_response_category?: string;
  risk_change_pts: number | null;
  verification_status: string;
  verification_message: string;
  audit_events?: DefenderAuditEvent[];
  pre_response_forecast?: ForecastResponse | null;
  post_response_forecast?: ForecastResponse | null;
  pre_response_explain?: ExplainResponse | null;
  post_response_explain?: ExplainResponse | null;
}

export interface DefenderApplyPayload {
  session_id: string;
  action: string;
  target: string;
  operator_id: string;
  authorization_reason: string;
  authorized: boolean;
}

export interface DefenderApplyResult {
  success: boolean;
  session_id: string;
  response_status: string;
  verification_status: string;
  verification_message: string;
  pre_response_risk: number;
  post_response_risk: number;
  risk_change_pts: number;
  flows_blocked_count: number;
  post_response_flows_count: number;
  session: any;
}

