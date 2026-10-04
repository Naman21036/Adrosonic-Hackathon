export type RetrievalMode = "dense" | "hybrid" | "bm25";

export interface BackendHealth {
  status?: string;
  embedding_model_ready?: boolean;
  qdrant_ready?: boolean;
  sparse_ready?: boolean;
  azure_ready?: boolean;
}

export type ConnectivityState = "CHECKING" | "ONLINE" | "OFFLINE" | "TIMEOUT" | "ERROR";

export interface SystemHealth extends BackendHealth {
  connectivity: ConnectivityState;
  endpoint: string | null;
  error: string | null;
  http_status: number | null;
  checked_at: string;
}

export interface BackendStatus {
  indexed_passages?: number;
  sparse_passages?: number;
  source_query_count?: number;
  collection?: string;
  dimension?: number | null;
  pending_mutations?: number;
  sparse_ready?: boolean;
  model?: string;
  sync_status?: string;
  data_source?: "live" | "validated_snapshot";
  snapshot_as_of?: string | null;
  validation_error?: string | null;
}

export interface IndexStats {
  qdrant_count: number | null;
  sparse_count: number | null;
  qdrant_unique_ids: number | null;
  sparse_unique_ids: number | null;
  source_query_count: number | null;
  dimension: number | null;
  model: string | null;
  collection: string | null;
  ids_match: boolean | null;
  snapshot_as_of: string | null;
  validation_status: string | null;
  validation_error: string | null;
}

export interface RetrievedPassage {
  id: string;
  text: string;
  score: number;
  source_id?: number | null;
  metadata?: Record<string, unknown>;
  retrieval_scores?: Record<string, number | undefined>;
  retrieval_mode?: string;
  score_type?: string;
}

export interface QueryRequest {
  query: string;
  mode: "dense" | "hybrid";
  top_k: number;
  filter?: { source_id: number };
  generate_answer?: boolean;
}

export interface LiveMutationResponse {
  success: boolean;
  operation: "upsert" | "delete";
  status: "created" | "updated" | "deleted";
  passage_id: string;
  source_id?: number | string | null;
  created?: boolean;
  updated?: boolean;
  embedding_dimension?: number;
  embedding_count_delta: number;
  indexed_passages_before: number;
  indexed_passages_after: number;
  dense_count_delta: number;
  sparse_count_before: number;
  sparse_count_after: number;
  qdrant_updated?: boolean;
  bm25_updated?: boolean;
  deleted_from_qdrant?: boolean;
  deleted_from_bm25?: boolean;
  latency_ms: number;
  stage_latency_ms?: Record<string, number>;
  pending_mutations: number;
}

export interface QueryResponse {
  query: string;
  mode: "dense" | "hybrid";
  top_k: number;
  filter?: Record<string, unknown>;
  latency_ms: number;
  retrieval_latency_ms?: number;
  total_latency_ms?: number;
  stage_latency_ms?: Record<string, number>;
  results: RetrievedPassage[];
  answer?: { answer?: string; generation_latency_ms?: number };
}

export interface EvaluationModeResult {
  retrieval_successes?: number | null;
  retrieval_failures?: number | null;
  context_precision?: number | null;
  context_recall?: number | null;
  evaluation_successes?: number | null;
  evaluation_failures?: number | null;
  status?: string;
}

export interface RetrievalModeMetrics {
  hit_at_1?: number | null;
  hit_at_5?: number | null;
  hit_at_10?: number | null;
  mrr?: number | null;
  queries_with_ground_truth_ranked?: number | null;
}

export interface EvaluationRow {
  queryId: string;
  mode: string;
  query: string;
  reference: string | null;
  passageIds: string[];
  errors: string[];
  contextPrecision: number | null;
  contextRecall: number | null;
}

export interface EvaluationReport {
  status: string;
  evaluationType: string | null;
  completedAt: string | null;
  datasetSource: string | null;
  datasetClassification: string | null;
  officialMsMarco: boolean | null;
  queryCount: number | null;
  validQueryCount: number | null;
  judgeProvider: string | null;
  judgeModel: string | null;
  ragasVersion: string | null;
  judgeEnabled: boolean | null;
  judgePreflightPassed: boolean | null;
  retrievalConfig: Record<string, number>;
  perModeResults: Record<string, EvaluationModeResult>;
  retrievalMetrics: Record<string, RetrievalModeMetrics>;
  aggregate: Record<string, EvaluationModeResult>;
  evaluationError: string | null;
  rows: EvaluationRow[];
  totalRows: number;
  offset: number;
  limit: number;
}

export interface LatencySummary {
  count?: number;
  successful?: number;
  failures?: number;
  mean_ms?: number | null;
  min_ms?: number | null;
  max_ms?: number | null;
  p50_ms?: number | null;
  p95_ms?: number | null;
  p99_ms?: number | null;
}

export interface BenchmarkSample {
  mode: string;
  query_number: number;
  latency_ms: number;
  error: string | null;
}

export interface BenchmarkReport {
  status: string;
  queryCount: number | null;
  requestedQueries: number | null;
  querySource: string | null;
  retrievalConfig: Record<string, number>;
  latency: Record<string, LatencySummary>;
  environment: Record<string, string | number | null>;
  samples: BenchmarkSample[];
}

export type DemoBenchmarkMode = "dense" | "bm25" | "hybrid";

export interface SimulatedBenchmarkRecord {
  query_id: string;
  mode: DemoBenchmarkMode;
  latency_ms: number;
  embedding_latency_ms: number;
  dense_latency_ms: number;
  bm25_latency_ms: number;
  rrf_latency_ms: number;
  total_latency_ms: number;
  status: "SIMULATED";
  executed: false;
}

export interface DemoLatencyBucket {
  label: string;
  min_ms: number;
  max_ms: number | null;
  count: number;
}

export interface DemoBenchmarkReport {
  source: "DEMO / SIMULATED";
  status: "SIMULATED";
  executed: false;
  generation_version: string;
  random_seed: number;
  query_count: number;
  mode_count: number;
  unique_query_ids: number;
  records_per_mode: number;
  mode_record_count: number;
  modes: DemoBenchmarkMode[];
  datasets: Record<DemoBenchmarkMode, SimulatedBenchmarkRecord[]>;
  summary: Record<DemoBenchmarkMode, LatencySummary>;
  latency_distribution: Record<DemoBenchmarkMode, DemoLatencyBucket[]>;
  notice: string;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  requestId: string;
  event: string;
  latencyMs: number | null;
  status: string;
}

export interface ObservabilityReport {
  logs: LogEntry[];
  logCount: number;
  langfuseEnabled: boolean | null;
  langfuseCredentialsPresent: boolean | null;
  langfuseHost: string | null;
  langfuseTracesAvailable: false;
  langfuseMissingEndpoint: string;
}

export interface FilterDemoReport {
  status: string;
  sourceId: number | null;
  sourcePassages: number | null;
  payloadIndexCreated: boolean;
  payloadIndexLimitation: string | null;
}

export interface ApiErrorBody {
  detail?: string;
}
