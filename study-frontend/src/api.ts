// Study API client (v2 — multi-study). Participant calls are same-origin; admin
// calls carry the X-Study-Admin-Token header.

import { traceHeaders, startTrace } from "@/lib/trace";

const BASE = "/api/study";

async function asError(r: Response): Promise<Error> {
  let detail = `HTTP ${r.status}`;
  try { const d = await r.json(); if (d?.detail) detail = d.detail; } catch { /* ignore */ }
  const e = new Error(detail) as Error & { status?: number };
  e.status = r.status;
  return e;
}

async function jpost(path: string, body: unknown, headers: Record<string, string> = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json", ...traceHeaders(), ...headers }, body: JSON.stringify(body),
  });
  if (!r.ok) throw await asError(r);
  return r.json();
}

async function jput(path: string, body: unknown, headers: Record<string, string> = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method: "PUT", headers: { "Content-Type": "application/json", ...traceHeaders(), ...headers }, body: JSON.stringify(body),
  });
  if (!r.ok) throw await asError(r);
  return r.json();
}

async function jget(path: string, headers: Record<string, string> = {}) {
  const r = await fetch(`${BASE}${path}`, { headers: { ...traceHeaders(), ...headers } });
  if (!r.ok) throw await asError(r);
  return r.json();
}

async function jdel(path: string, headers: Record<string, string> = {}) {
  const r = await fetch(`${BASE}${path}`, { method: "DELETE", headers: { ...traceHeaders(), ...headers } });
  if (!r.ok) throw await asError(r);
  return r.json();
}

// ---------- participant ----------
export interface ScenarioInfo {
  scenario_order: number;
  scenario_id: number;
  title: string;
  role: string;
  current_situation: string;
  goal: string;
  how_to_interact: string;
  suggested_first_line: string;
  opening_details?: string;
  additional_details: string;
  final_account_prompt?: string;
  extra_fields?: { label: string; value: string }[];
  post_items?: any[];
  time_limit_s: number;
  study_role?: "practice" | "analytical";
}
export interface RunState {
  status: "not_started" | "in_progress" | "submitted" | "expired";
  current_step?: Record<string, any>;
  completed?: Record<string, any>;
  remaining_seconds?: number;
  attempt?: number;
}
export interface EnterResult {
  participant_id: string;
  study_name: string;
  scenarios: ScenarioInfo[];
  questionnaires?: Record<string, any[]>;
  welcome_text?: string;
  estimated_duration?: string;
  practice_intro_text?: string;
  main_intro_text?: string;
  run: RunState;
}
export interface PrepareState {
  status: "idle" | "preparing" | "ready" | "error";
  steps: { label: string; state: string }[];
  error?: string | null;
  version: number;
}

export const api = {
  enter: (code: string): Promise<EnterResult> => jpost("/enter", { code }),
  runStart: (code: string, mode: "resume" | "restart") => jpost("/run/start", { code, mode }),
  progress: (code: string, current_step: Record<string, any>, completed: Record<string, any>) =>
    jpost("/run/progress", { code, current_step, completed }),
  sessionStart: (code: string, scenario_order: number): Promise<{ session_id: string; scenario: ScenarioInfo; prepare: PrepareState }> => {
    startTrace(); // one trace per scenario: session/start + chat-proxy WS + save group together
    return jpost("/session/start", { code, scenario_order });
  },
  prepareStatus: (): Promise<PrepareState> => jget("/run/prepare/status"),
  sessionEnd: (sessionId: string, reason: string) => jpost(`/session/${sessionId}/end`, { reason }),
  questionnaire: (sessionId: string | null, code: string, kind: string, payload: Record<string, any>) =>
    jpost(`/session/${sessionId ?? "none"}/questionnaire`, { code, kind, payload }),
  submit: (code: string) => jpost("/run/submit", { code }),
  audioCheckStart: (code: string): Promise<{ session_id: string; prepare: PrepareState }> => {
    startTrace();
    return jpost("/audio-check/start", { code });
  },
  telemetry: (session_id: string, marks: Record<string, number>) =>
    jpost("/telemetry", { session_id, marks }).catch(() => {}),
  // Round-trip the /ping endpoint to estimate browser↔server network latency (ms).
  async pingRttMs(): Promise<number | null> {
    try {
      const t0 = performance.now();
      await fetch(`${BASE}/ping`, { headers: traceHeaders() });
      return performance.now() - t0;
    } catch { return null; }
  },
  playbackUrl: (code: string, scenarioOrder?: number, track?: string,
                condition?: string, maxDurationS?: number, clipIndex?: number) => {
    const qs = new URLSearchParams()
    if (scenarioOrder) qs.set("scenario", String(scenarioOrder))
    if (track) qs.set("track", track)
    if (condition) qs.set("condition", condition)
    if (maxDurationS) qs.set("max_duration_s", String(maxDurationS))
    if (clipIndex) qs.set("clip_index", String(clipIndex))
    const q = qs.toString()
    return `${BASE}/playback/${encodeURIComponent(code)}${q ? `?${q}` : ""}`
  },

  async saveSession(sessionId: string, arts: {
    participant?: Blob | null; participant_raw?: Blob | null; model?: Blob | null;
    merged?: Blob | null; model_transcript?: unknown; client_timeline?: unknown;
  }) {
    const fd = new FormData();
    if (arts.participant) fd.append("participant", arts.participant, "participant.wav");
    if (arts.participant_raw) fd.append("participant_raw", arts.participant_raw, "participant_raw.wav");
    if (arts.model) fd.append("model", arts.model, "model.wav");
    if (arts.merged) fd.append("merged", arts.merged, "merged.wav");
    // Timelines grow beyond Starlette's 1 MB limit for ordinary multipart text
    // fields during longer calls. A filename makes these proper spooled file parts.
    fd.append("model_transcript_file", new Blob(
      [JSON.stringify(arts.model_transcript ?? null)], { type: "application/json" }
    ), "model_transcript.json");
    fd.append("client_timeline_file", new Blob(
      [JSON.stringify(arts.client_timeline ?? null)], { type: "application/json" }
    ), "client_timeline.json");
    const r = await fetch(`${BASE}/session/${sessionId}/save`, { method: "POST", headers: traceHeaders(), body: fd });
    if (!r.ok) throw await asError(r);
    return r.json();
  },
};

// EventSource stream of prepare state; onState called on each update; resolves when terminal.
export function streamPrepare(onState: (s: PrepareState) => void): () => void {
  const es = new EventSource(`${BASE}/run/prepare/stream`);
  es.onmessage = (e) => {
    try {
      const s = JSON.parse(e.data) as PrepareState;
      onState(s);
      if (s.status === "ready" || s.status === "error") es.close();
    } catch { /* ignore */ }
  };
  es.onerror = () => es.close();
  return () => es.close();
}

// ---------- admin ----------
export function adminHeaders(token: string) { return { "X-Study-Admin-Token": token }; }

export const adminApi = {
  voices: (t: string): Promise<{ voices: string[] }> => jget("/voices", adminHeaders(t)),
  engines: (t: string): Promise<{ engines: string[] }> => jget("/engines", adminHeaders(t)),
  stopEngine: (t: string) => jpost("/stop-engine", {}, adminHeaders(t)),

  listStudies: (t: string) => jget("/studies", adminHeaders(t)),
  createStudy: (t: string, name: string, description = "") => jpost("/studies", { name, description }, adminHeaders(t)),
  getStudy: (t: string, id: number) => jget(`/studies/${id}`, adminHeaders(t)),
  updateStudy: (t: string, id: number, body: { name?: string; description?: string; settings?: any }) => jput(`/studies/${id}`, body, adminHeaders(t)),
  archiveStudy: (t: string, id: number) => jdel(`/studies/${id}`, adminHeaders(t)),
  releaseRun: (t: string, id: number, runId: number) =>
    jpost(`/studies/${id}/runs/${runId}/release`, {}, adminHeaders(t)),
  restoreStudy: (t: string, id: number) => jpost(`/studies/${id}/restore`, {}, adminHeaders(t)),
  setQuestionnaires: (t: string, id: number, questionnaires: unknown) =>
    jput(`/studies/${id}/questionnaires`, { questionnaires }, adminHeaders(t)),

  addScenario: (t: string, id: number, scenario: unknown) => jpost(`/studies/${id}/scenarios`, scenario, adminHeaders(t)),
  updateScenario: (t: string, id: number, sid: number, scenario: unknown) => jput(`/studies/${id}/scenarios/${sid}`, scenario, adminHeaders(t)),
  deleteScenario: (t: string, id: number, sid: number) => jdel(`/studies/${id}/scenarios/${sid}`, adminHeaders(t)),
  setScenarioPostItems: (t: string, id: number, sid: number, post_items: any[]) =>
    jput(`/studies/${id}/scenarios/${sid}/post-items`, { post_items }, adminHeaders(t)),

  async uploadTarget(t: string, id: number, ref: string, speakerId: string, label: string, engine: string, file: File) {
    const fd = new FormData();
    fd.append("wav", file); fd.append("ref", ref); fd.append("speaker_id", speakerId);
    fd.append("label", label); fd.append("engine", engine);
    const r = await fetch(`${BASE}/studies/${id}/targets`, { method: "POST", headers: adminHeaders(t), body: fd });
    if (!r.ok) throw await asError(r);
    return r.json();
  },
  deleteTarget: (t: string, id: number, tid: number) => jdel(`/studies/${id}/targets/${tid}`, adminHeaders(t)),

  generate: (t: string, id: number, count: number) => jpost(`/studies/${id}/participants/generate`, { count }, adminHeaders(t)),
  counterbalance: (t: string, id: number) => jget(`/studies/${id}/counterbalance`, adminHeaders(t)),
  runs: (t: string, id: number) => jget(`/studies/${id}/runs`, adminHeaders(t)),
  sessions: (t: string, id: number) => jget(`/studies/${id}/sessions`, adminHeaders(t)),
  analyze: (t: string, id: number, force = false) => jpost(`/studies/${id}/analyze${force ? "?force=true" : ""}`, {}, adminHeaders(t)),
  analyzeStatus: (t: string, id: number) => jget(`/studies/${id}/analyze/status`, adminHeaders(t)),
  vcQuality: (t: string, id: number, body: { participant_id?: string; session_id?: string; force?: boolean }) =>
    jpost(`/studies/${id}/vc-quality`, body, adminHeaders(t)),
  vcQualityStatus: (t: string, id: number) => jget(`/studies/${id}/vc-quality/status`, adminHeaders(t)),
  exportUrl: (id: number, format: "json" | "zip") => `${BASE}/studies/${id}/export?format=${format}`,
  datasetUrl: (id: number) => `${BASE}/studies/${id}/dataset`,

  templateUrl: () => `${BASE}/template`,
  yamlUrl: (id: number) => `${BASE}/studies/${id}/yaml`,
  async importYaml(t: string, id: number, file: File) {
    const fd = new FormData(); fd.append("file", file)
    const r = await fetch(`${BASE}/studies/${id}/import`, { method: "POST", headers: adminHeaders(t), body: fd })
    if (!r.ok) throw await asError(r)
    return r.json()
  },
  async download(t: string, url: string, filename: string) {
    const r = await fetch(url, { headers: adminHeaders(t) })
    if (!r.ok) throw await asError(r)
    const blob = await r.blob()
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob); a.download = filename; a.click(); URL.revokeObjectURL(a.href)
  },
};
