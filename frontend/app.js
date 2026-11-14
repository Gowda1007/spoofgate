/* SpoofGate kiosk UI. Plain JS, no build step. Served by FastAPI at "/". */
'use strict';

// ------------------------------------------------------------------ config
const API = new URLSearchParams(location.search).get('api') || '';   // same origin by default
const KIOSK_ID = 'kiosk-01';
const BURST = { n: 6, gapMs: 80 };            // ~0.4 s capture
const ACTION_BURST = { n: 8, gapMs: 150 };    // head-turn step-up, ~1.1 s
const ENROLL_BURST = { n: 8, gapMs: 120 };
const SEND_WIDTH = 640;
const JPEG_Q = 0.85;
const VERIFY_TIMEOUT_MS = 6000;
const HEALTH_EVERY_MS = 4000;
const SLA_MS = 2000;
const ID_UNSURE = 0.25;                        // matches app/config.py ID_UNSURE
const OFFLINE = { minLive: 0.85, minSim: 0.45, restricted: ['server_room'], maxAgeH: 72 };
const QUEUE_KEY = 'spoofgate.offlineQueue.v1';

const ZONES = {
  campus_gate: 'Main Campus Gate',
  lab_cse: 'AI Lab (CSE)',
  server_room: 'Server Room',
};

const REASONS = {
  LIVE_MATCH: 'Live person, identity verified',
  AUTHORIZED_24_7_CAMPUS_ACCESS: 'Live match · zone open 24/7',
  SCHEDULE_AUTHORIZED: 'Live match · inside access schedule',
  SPOOF_DETECTED: 'Presentation attack detected',
  NO_MATCH: 'Face is not enrolled',
  LOW_QUALITY: 'Face not clear enough',
  LOW_CONFIDENCE: 'Low confidence, extra check needed',
  LOCKED_OUT: 'Too many failed attempts, kiosk locked for a few minutes',
  TOO_MANY_RETRIES: 'Too many unclear attempts',
  OTP_VERIFIED: 'One-time code verified',
  OTP_WRONG: 'Wrong code, try again',
  OTP_FAILED: 'Code failed 3 times',
  ACTION_VERIFIED: 'Head-turn check passed',
  ACTION_FAILED: 'Head-turn check failed',
  USER_NOT_FOUND_IN_POLICY: 'Person has no campus access record',
  ACCOUNT_REVOKED_OR_SUSPENDED: 'Access revoked or suspended',
  UNAUTHORIZED_ZONE: 'No clearance for this zone',
  RESTRICTED_FACULTY_ZONE: 'Restricted zone · faculty only',
  AFTER_HOURS_POLICY_TRIGGER: 'Outside access schedule, approval code needed',
  OFFLINE_RESTRICTED_ZONE: 'Offline: restricted zones stay closed',
  OFFLINE_STRICT_STEPUP: 'Offline: stricter bar, local head-turn check needed',
  OFFLINE_ID_UNCERTAIN: 'Offline: identity not certain, please see security desk',
  OFFLINE_NEEDS_APPROVAL: 'Offline: after-hours entry needs online approval',
  ENGINE_UNREACHABLE: 'Verification engine unreachable · no entry (fail closed)',
  ENGINE_TIMEOUT: 'Decision took too long · no entry (fail closed)',
  ENGINE_ERROR: 'Engine error · no entry (fail closed)',
  CAMERA_ERROR: 'Camera not ready',
};
const QUALITY_HINTS = {
  no_face: 'No face found, step into the oval',
  face_too_small: 'Move closer to the camera',
  too_dark: 'Too dark, face the light',
  too_bright: 'Too bright, avoid direct light',
  blurry: 'Hold still for a moment',
  head_turned: 'Look straight at the camera',
  low_quality: 'Image unclear, try again',
};

// ------------------------------------------------------------------ state
const S = {
  health: null, engineUp: null, offline: false, busy: false, timeOverride: '',
  session: null, last: null, lastCtx: null, audit: [], queue: loadQueue(), stream: null,
  t0: 0, raf: 0,
};

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtMs = (ms) => (ms == null || isNaN(ms) ? '—' : ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms');
const pct = (x) => (x == null ? '—' : Math.round(x * 100) + '%');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
const normDecision = (d) => (d === 'ALLOW' ? 'ALLOW' : d === 'DENY' ? 'DENY' : 'STEP_UP'); // REQUEST_VERIFICATION | STEP_UP -> STEP_UP

function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg; t.classList.add('on');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('on'), ms);
}

async function api(path, { method = 'GET', body, timeout = 5000 } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const res = await fetch(API + path, {
      method, signal: ctl.signal,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      const err = new Error((data && (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))) || res.statusText);
      err.status = res.status; throw err;
    }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') { const err = new Error('timeout'); err.timeout = true; throw err; }
    throw e;
  } finally { clearTimeout(timer); }
}

// ------------------------------------------------------------------ camera
async function startCamera() {
  const msg = $('cam-msg');
  msg.hidden = true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return camError('Camera blocked', 'Browsers only allow the camera on https:// or http://localhost. Open the app from localhost or the https tunnel URL.');
  }
  try {
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
    });
    const v = $('video');
    v.srcObject = S.stream;
    await v.play();
    const s = S.stream.getVideoTracks()[0];
    $('cam-meta').textContent = `${v.videoWidth}×${v.videoHeight} · ${s.label || 'camera'}`;
    // A virtual-camera name is only a weak hint (it can be renamed), so we flag it, we don't block on it.
    if (/obs|virtual|manycam|snap camera|xsplit/i.test(s.label || '')) {
      toast('Warning: virtual camera software detected');
      $('cam-meta').textContent += ' · ⚠ virtual camera';
    }
    updateVerifyButton();
  } catch (e) {
    const why = e.name === 'NotAllowedError' ? 'Permission denied. Allow camera access in the browser address bar, then retry.'
      : e.name === 'NotFoundError' ? 'No camera found. Plug in a webcam, then retry.'
      : e.name === 'NotReadableError' ? 'Camera is in use by another app (Zoom, Teams, tools.py collect). Close it, then retry.'
      : e.message;
    camError('Camera unavailable', why);
  }
}
function camError(title, text) {
  $('cam-msg-title').textContent = title;
  $('cam-msg-text').textContent = text;
  $('cam-msg').hidden = false;
  $('cam-meta').textContent = 'no camera';
  updateVerifyButton();
}
const camReady = () => { const v = $('video'); return !!(S.stream && v.videoWidth); };

async function captureBurst({ n, gapMs }) {
  const v = $('video');
  if (!camReady()) throw Object.assign(new Error('Camera not ready'), { camera: true });
  const scale = Math.min(1, SEND_WIDTH / v.videoWidth);
  const c = document.createElement('canvas');
  c.width = Math.round(v.videoWidth * scale);
  c.height = Math.round(v.videoHeight * scale);
  const ctx = c.getContext('2d');
  const frames = [];
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    ctx.drawImage(v, 0, 0, c.width, c.height);
    frames.push(c.toDataURL('image/jpeg', JPEG_Q));
    $('flash').classList.add('on'); setTimeout(() => $('flash').classList.remove('on'), 60);
    if (i < n - 1) await sleep(gapMs);
  }
  return { frames, captureMs: performance.now() - t0 };
}

// ------------------------------------------------------------------ health / status
async function heartbeat() {
  try {
    S.health = await api('/api/health', { timeout: 2500 });
    S.engineUp = true;
  } catch {
    S.engineUp = false;
  }
  renderStatus();
}

function setPill(id, cls, text) {
  const p = $(id);
  p.className = 'pill ' + cls;
  p.querySelector('span').textContent = text;
}

function renderStatus() {
  const h = S.health || {};
  if (S.engineUp === false) setPill('pill-engine', 'bad', 'unreachable · fail closed');
  else if (S.engineUp) setPill('pill-engine', h.detector && /yunet/i.test(h.detector) ? 'ok' : 'warn', `online · ${h.detector || '?'}`);
  if (S.engineUp) {
    setPill('pill-model', h.liveness_trained ? 'ok' : 'warn', h.liveness_trained ? 'trained' : 'NOT trained (fail-closed)');
  }
  setPill('pill-link', S.offline ? 'warn' : 'ok', S.offline ? 'OFFLINE · safe mode' : 'online');
  setPill('pill-clock', S.timeOverride ? 'warn' : 'ok', S.timeOverride ? `DEMO OVERRIDE ${S.timeOverride}` : 'live (server)');

  // admin health
  const kv = $('health-kv');
  if (kv) {
    const rows = S.engineUp ? [
      ['Detector', h.detector], ['Face model (SFace)', h.identity_model ? 'loaded' : 'MISSING'],
      ['MiniFASNet ONNX', h.antispoof_onnx ? 'loaded' : 'not used'], ['Liveness classifier', h.liveness_trained ? 'trained' : 'not trained (fail-closed)'],
      ['AWS Rekognition', h.aws_rekognition_available ? 'Active (ap-south-1)' : 'Local Zero-Cost Mode'],
      ['Cost Pre-filter', h.cost_prefilter_active ? 'Active (saves $0.015/call)' : 'disabled'],
      ['Deny below / allow above', `${h.t_deny?.toFixed?.(2)} / ${h.t_allow?.toFixed?.(2)}`], ['Match threshold', h.id_match],
      ['Enrolled', (h.enrolled_users || []).join(', ') || 'none'],
    ] : [['Engine', 'unreachable']];
    kv.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    const sel = $('revoke-id');
    const users = h.enrolled_users || [];
    const cur = sel.value;
    sel.innerHTML = users.length ? users.map((u) => `<option>${esc(u)}</option>`).join('') : '<option value="">no users</option>';
    if (users.includes(cur)) sel.value = cur;
  }
  // threshold notes
  if (h.t_deny != null) $('live-note').textContent = `Deny below ${h.t_deny.toFixed(2)} · allow above ${h.t_allow.toFixed(2)}`;
  setMarkers();
  updateVerifyButton();
}

function updateVerifyButton() {
  const b = $('btn-verify');
  const ok = S.engineUp && camReady() && !S.busy;
  b.disabled = !ok;
  $('verify-text').textContent = S.busy ? 'Verifying…' : S.engineUp === false ? 'Engine offline · no entry' : !camReady() ? 'Waiting for camera' : 'Verify entry';
}

// ------------------------------------------------------------------ verify flow
function contextPayload() {
  const ctx = { zone: $('zone').value, is_offline: S.offline };
  if (S.timeOverride) ctx.time_str = S.timeOverride;   // demo only; otherwise server clock is used
  return ctx;
}

function startWatch() {
  S.t0 = performance.now();
  const tick = () => { $('watch').textContent = fmtMs(performance.now() - S.t0); S.raf = requestAnimationFrame(tick); };
  cancelAnimationFrame(S.raf); tick();
}
function stopWatch() { cancelAnimationFrame(S.raf); const t = performance.now() - S.t0; $('watch').textContent = fmtMs(t); return t; }

function setBusy(on) {
  S.busy = on;
  $('cam').classList.toggle('busy', on);
  updateVerifyButton();
}

async function verify({ reuseSession = false } = {}) {
  if (S.busy) return;
  if (!S.engineUp) { showFailClosed('ENGINE_UNREACHABLE'); return; }
  if (!reuseSession || !S.session) S.session = 'k-' + uid().slice(0, 12);
  hideStep();
  setBusy(true);
  setHero('busy', 'CAPTURING', 'Hold still', 'Taking 6 frames…');
  resetTrace(); markTrace('capture', 'run');
  startWatch();
  const ctx = contextPayload();
  let cap;
  try {
    cap = await captureBurst(BURST);
  } catch (e) {
    stopWatch(); setBusy(false);
    showResult(failClosed('CAMERA_ERROR', e.message), {});
    return;
  }
  setHero('busy', 'ANALYSING', 'Checking liveness and identity', 'On-device CPU pipeline');
  try {
    const r = await api('/api/verify', {
      method: 'POST', timeout: VERIFY_TIMEOUT_MS,
      body: {
        frames: cap.frames, session_id: S.session, scenario_label: $('label').value || null,
        claimed_id: $('claimed').value.trim() || null, device_id: KIOSK_ID,
        capture_ms: Math.round(cap.captureMs), context: ctx,
      },
    });
    const total = stopWatch();
    const final = applyOfflineRules(r, ctx.zone);
    showResult(final, { total, capture: cap.captureMs, ctx });
  } catch (e) {
    stopWatch();
    const code = e.timeout ? 'ENGINE_TIMEOUT' : e.status ? 'ENGINE_ERROR' : 'ENGINE_UNREACHABLE';
    showResult(failClosed(code, e.message), { ctx });
    if (!e.status) heartbeat();
  } finally {
    setBusy(false);
    refreshStats();
  }
}

// Never fail open: any error becomes a DENY with a clear reason.
function failClosed(code, detail) {
  return { decision: 'DENY', reason_code: code, detail, identity: {}, timings_ms: {}, _local: true };
}
function showFailClosed(code) { showResult(failClosed(code), { ctx: contextPayload() }); }

/* Offline safe mode. The kiosk still decides locally, but it raises the bar and never
   auto-opens on doubt. (Mirror these rules server-side using context.is_offline.) */
function applyOfflineRules(r, zone, { fromStep = false } = {}) {
  const out = { ...r, decision: normDecision(r.decision), offline_rule: null };
  if (!S.offline) return out;
  const live = r.liveness_score, sim = r.identity?.similarity;
  const idMatch = S.health?.id_match ?? 0.363;
  const deny = (code, rule) => ({ ...out, decision: 'DENY', reason_code: code, next_step: null, offline_rule: rule });

  if (OFFLINE.restricted.includes(zone) && out.decision !== 'DENY') return deny('OFFLINE_RESTRICTED_ZONE', 'Restricted zone closed offline');
  if (out.decision === 'DENY') return { ...out, offline_rule: 'Deny kept' };

  if (fromStep) {
    // after the local head-turn fallback, identity must still be a full match
    if (out.decision === 'ALLOW' && !(sim != null && sim >= idMatch)) return deny('OFFLINE_ID_UNCERTAIN', 'Fallback passed but identity not certain');
    return { ...out, offline_rule: 'Local fallback accepted' };
  }
  if (out.decision === 'ALLOW') {
    if (!(live >= OFFLINE.minLive && sim >= OFFLINE.minSim)) {
      return { ...out, decision: 'STEP_UP', reason_code: 'OFFLINE_STRICT_STEPUP', next_step: 'ACTION', offline_rule: `Below offline bar (live ≥ ${OFFLINE.minLive}, match ≥ ${OFFLINE.minSim})` };
    }
    return { ...out, offline_rule: 'Passed stricter offline bar' };
  }
  // STEP_UP cases
  if (r.context_eval || r.reason_code === 'AFTER_HOURS_POLICY_TRIGGER') return deny('OFFLINE_NEEDS_APPROVAL', 'Schedule exceptions need the central service');
  if (out.next_step === 'OTP') return deny('OFFLINE_ID_UNCERTAIN', 'OTP needs the network; identity doubt → security desk');
  if (out.next_step === 'RECAPTURE') return { ...out, offline_rule: 'Recapture allowed' };
  return { ...out, next_step: 'ACTION', offline_rule: 'Liveness doubt → local head-turn fallback' };
}

// ------------------------------------------------------------------ rendering
function setHero(cls, kicker, title, sub, code = '') {
  const h = $('hero');
  h.className = 'hero ' + cls;
  $('hero-kicker').textContent = kicker;
  $('hero-title').textContent = title;
  $('hero-sub').textContent = sub;
  $('hero-code').textContent = code;
  $('hero-icon').textContent = cls === 'allow' ? '✓' : cls === 'deny' ? '✕' : cls === 'step' ? '!' : cls === 'busy' ? '…' : '○';
  const cam = $('cam');
  cam.classList.remove('r-allow', 'r-step', 'r-deny');
  if (cls === 'allow' || cls === 'step' || cls === 'deny') cam.classList.add('r-' + cls);
}

function reasonText(r) {
  let t = REASONS[r.reason_code] || r.reason_code || '';
  if (r.reason_code === 'LOW_QUALITY' && r.reasons?.length) t = r.reasons.map((x) => QUALITY_HINTS[x] || x).join(' · ');
  if (r.reason_code === 'UNAUTHORIZED_ZONE' && r.context_eval?.reason) t = r.context_eval.reason;
  if (r.detail && r._local) t += ` (${r.detail})`;
  return t;
}

function showResult(r, meta) {
  const d = normDecision(r.decision);
  S.last = r; S.lastCtx = meta.ctx || S.lastCtx;
  const zoneName = ZONES[meta.ctx?.zone] || '';
  if (d === 'ALLOW') setHero('allow', 'ALLOW', 'Access granted', reasonText(r) + (zoneName ? ` · ${zoneName}` : ''), r.reason_code);
  else if (d === 'DENY') setHero('deny', 'DENY', r.reason_code === 'LOCKED_OUT' ? 'Kiosk locked' : 'Access denied', reasonText(r), r.reason_code);
  else setHero('step', 'STEP-UP VERIFICATION', 'Extra check needed', reasonText(r), r.reason_code);

  renderTrace(r, d);
  renderTiles(r, meta);
  if (d === 'STEP_UP' && r.next_step) showStep(r);
  record(r, d, meta);
}

function resetTrace() { document.querySelectorAll('#trace li').forEach((li) => { li.className = ''; li.querySelector('i').textContent = ''; }); }
function markTrace(k, cls) {
  const li = document.querySelector(`#trace li[data-k="${k}"]`);
  if (!li) return;
  li.className = cls === 'run' ? '' : cls;
  li.querySelector('i').textContent = cls === 'ok' ? '✓' : cls === 'bad' ? '✕' : cls === 'warn' ? '!' : cls === 'skip' ? '–' : '';
}

function renderTrace(r, d) {
  const h = S.health || {};
  const code = r.reason_code;
  const engineFail = r._local;
  markTrace('capture', code === 'CAMERA_ERROR' ? 'bad' : 'ok');
  if (engineFail || code === 'LOCKED_OUT') {
    ['quality', 'liveness', 'identity', 'policy'].forEach((k) => markTrace(k, code === 'CAMERA_ERROR' ? 'skip' : 'skip'));
  } else {
    const qBad = code === 'LOW_QUALITY' || code === 'TOO_MANY_RETRIES';
    markTrace('quality', qBad ? (code === 'LOW_QUALITY' ? 'warn' : 'bad') : 'ok');
    const p = r.liveness_score;
    if (p == null) markTrace('liveness', 'skip');
    else markTrace('liveness', p < (h.t_deny ?? 0.3) ? 'bad' : p >= (h.t_allow ?? 0.7) ? 'ok' : 'warn');
    const sim = r.identity?.similarity;
    if (sim == null) markTrace('identity', 'skip');
    else markTrace('identity', r.identity.matched ? 'ok' : sim >= ID_UNSURE ? 'warn' : 'bad');
    if (r.context_eval) {
      const policyDeny = ['USER_NOT_FOUND_IN_POLICY', 'ACCOUNT_REVOKED_OR_SUSPENDED', 'UNAUTHORIZED_ZONE', 'RESTRICTED_FACULTY_ZONE'].includes(code);
      markTrace('policy', policyDeny ? 'bad' : code === 'AFTER_HOURS_POLICY_TRIGGER' ? 'warn' : 'ok');
    } else markTrace('policy', 'skip');
  }
  if (S.offline) markTrace('offline', !r.offline_rule ? 'skip' : d === 'DENY' ? 'bad' : d === 'STEP_UP' ? 'warn' : 'ok');
  markTrace('decision', d === 'ALLOW' ? 'ok' : d === 'DENY' ? 'bad' : 'warn');
}

function setMeter(id, value, a, b, colorFn) {
  const m = $(id);
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  m.querySelector('.fill').style.width = (v * 100) + '%';
  m.querySelector('.fill').style.background = value == null ? 'var(--muted)' : colorFn(value);
  if (a != null) m.querySelector('.mk.a').style.left = `calc(${a * 100}% - 1px)`;
  if (b != null) m.querySelector('.mk.b').style.left = `calc(${b * 100}% - 1px)`;
}
function setMarkers() {
  const h = S.health || {};
  setMeter('live-meter', S.last?.liveness_score ?? null, h.t_deny ?? 0.3, h.t_allow ?? 0.7, liveColor);
  setMeter('id-meter', S.last?.identity?.similarity ?? null, ID_UNSURE, h.id_match ?? 0.363, idColor);
}
function liveColor(p) { const h = S.health || {}; return p < (h.t_deny ?? 0.3) ? 'var(--red)' : p >= (h.t_allow ?? 0.7) ? 'var(--green)' : 'var(--amber)'; }
function idColor(s) { const h = S.health || {}; return s >= (h.id_match ?? 0.363) ? 'var(--green)' : s >= ID_UNSURE ? 'var(--amber)' : 'var(--red)'; }
function setChip(id, cls, text) { const c = $(id); c.className = 'chip ' + (cls || ''); c.textContent = text; }

function renderTiles(r, meta) {
  // Latency
  const t = r.timings_ms || {};
  const total = meta.total;
  const server = t.total_server;
  const capture = meta.capture;
  const network = total != null && server != null && capture != null ? Math.max(0, total - capture - server) : null;
  $('lat-total').textContent = fmtMs(total);
  const within = total != null && total < SLA_MS;
  setChip('lat-chip', total == null ? '' : within ? 'ok' : 'bad', total == null ? '—' : within ? 'within 2 s' : 'over 2 s');
  const fill = $('sla-fill');
  fill.style.width = total == null ? '0' : Math.min(100, (total / SLA_MS) * 100) + '%';
  fill.style.background = within ? 'var(--green)' : 'var(--red)';
  const rows = [['Capture (6 frames)', capture], ['Network', network], ['Server total', server],
    ['· decode', t.decode], ['· detect', t.detect], ['· liveness', t.liveness], ['· identity', t.identity]].filter(([, v]) => v != null);
  $('lat-break').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${fmtMs(v)}</dd>`).join('');

  // Liveness
  const p = r.liveness_score;
  $('live-val').textContent = p == null ? '—' : pct(p);
  const h = S.health || {};
  setChip('live-chip', p == null ? '' : p < (h.t_deny ?? 0.3) ? 'bad' : p >= (h.t_allow ?? 0.7) ? 'ok' : 'warn',
    p == null ? 'not scored' : p < (h.t_deny ?? 0.3) ? 'spoof' : p >= (h.t_allow ?? 0.7) ? 'live' : 'uncertain');
  setMeter('live-meter', p, h.t_deny ?? 0.3, h.t_allow ?? 0.7, liveColor);

  // Identity
  const id = r.identity || {};
  $('id-user').textContent = id.user_id ? id.user_id : id.similarity != null ? 'Unknown' : '—';
  setChip('id-chip', id.similarity == null ? '' : id.matched ? 'ok' : id.similarity >= ID_UNSURE ? 'warn' : 'bad',
    id.similarity == null ? 'not checked' : `${id.matched ? 'match' : id.similarity >= ID_UNSURE ? 'unsure' : 'no match'} · ${id.similarity.toFixed(2)}`);
  setMeter('id-meter', id.similarity ?? null, ID_UNSURE, h.id_match ?? 0.363, idColor);

  // Policy
  const ce = r.context_eval;
  const ctx = meta.ctx || {};
  $('pol-zone').textContent = ZONES[ctx.zone] || ctx.zone || '—';
  const kv = [];
  kv.push(['Clock', ctx.time_str ? `${ctx.time_str} (demo override)` : 'server clock']);
  if (ce) {
    if (ce.window) kv.push(['Window', ce.window]);
    if (ce.schedule) kv.push(['Schedule', ce.schedule === '24_7' ? 'open 24/7' : ce.schedule]);
    if (ce.user_role) kv.push(['Role', ce.user_role]);
    if (ce.status) kv.push(['Status', ce.status]);
  }
  if (r.offline_rule) kv.push(['Offline rule', r.offline_rule]);
  $('pol-kv').innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  const code = r.reason_code;
  if (ce) {
    const bad = ['USER_NOT_FOUND_IN_POLICY', 'ACCOUNT_REVOKED_OR_SUSPENDED', 'UNAUTHORIZED_ZONE', 'RESTRICTED_FACULTY_ZONE', 'OFFLINE_RESTRICTED_ZONE', 'OFFLINE_NEEDS_APPROVAL'].includes(code);
    setChip('pol-chip', bad ? 'bad' : normDecision(r.decision) === 'STEP_UP' ? 'warn' : 'ok', bad ? 'not authorised' : code === 'AFTER_HOURS_POLICY_TRIGGER' ? 'after hours' : 'authorised');
  } else setChip('pol-chip', '', 'checked after biometrics');
}

// ------------------------------------------------------------------ step-up
function hideStep() { $('step').hidden = true; $('step-err').textContent = ''; $('otp-demo').textContent = ''; }
function showStep(r) {
  const s = r.next_step;
  $('step').hidden = false;
  $('step-otp').hidden = s !== 'OTP';
  $('step-action').hidden = s !== 'ACTION';
  $('step-recapture').hidden = s !== 'RECAPTURE';
  $('step-err').textContent = '';
  if (s === 'OTP') {
    $('step-title').textContent = 'Enter the one-time code';
    $('step-text').textContent = 'A 6-digit code was sent to the registered phone. Three wrong codes deny entry.';
    $('otp').value = ''; setTimeout(() => $('otp').focus(), 50);
  } else if (s === 'ACTION') {
    $('step-title').textContent = 'Head-turn check';
    $('step-text').textContent = 'Press start, then slowly turn your head left and right for about one second.';
  } else {
    $('step-title').textContent = 'Let’s try that again';
    $('step-text').textContent = reasonText(r) || 'Face the camera inside the oval.';
  }
}

async function doStep(step, extra) {
  if (S.busy || !S.session) return;
  setBusy(true);
  $('step-err').textContent = '';
  startWatch();
  try {
    let frames;
    if (step === 'ACTION') {
      setHero('busy', 'HEAD-TURN CHECK', 'Turn your head slowly', 'left … and right');
      frames = (await captureBurst(ACTION_BURST)).frames;
    }
    const r = await api('/api/verify-step', {
      method: 'POST', timeout: VERIFY_TIMEOUT_MS,
      body: { session_id: S.session, step, otp: extra?.otp, frames, device_id: KIOSK_ID },
    });
    const total = stopWatch();
    const merged = { ...r, identity: r.identity || S.last?.identity, liveness_score: r.liveness_score ?? S.last?.liveness_score };
    const final = applyOfflineRules(merged, S.lastCtx?.zone, { fromStep: true });
    hideStep();
    showResult(final, { total, ctx: S.lastCtx });
  } catch (e) {
    stopWatch();
    $('step-err').textContent = e.timeout ? 'Timed out, no entry. Try again.' : `Could not verify: ${e.message}`;
    if (S.last) showStep(S.last);
    $('step-err').textContent = e.timeout ? 'Timed out, no entry. Try again.' : `Could not verify: ${e.message}`;
  } finally {
    setBusy(false);
    refreshStats();
  }
}

// ------------------------------------------------------------------ audit + offline queue
function loadQueue() {
  try {
    const q = JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]');
    const cutoff = Date.now() - OFFLINE.maxAgeH * 3600e3;   // auto-expire old records
    return q.filter((e) => Date.parse(e.ts) > cutoff);
  } catch { return []; }
}
function saveQueue() { try { localStorage.setItem(QUEUE_KEY, JSON.stringify(S.queue)); } catch { /* storage unavailable */ } }

function record(r, d, meta) {
  // Minimum data only: no images, no templates, no embeddings.
  const ev = {
    event_id: uid(), ts: new Date().toISOString(), kiosk_id: KIOSK_ID, zone: meta.ctx?.zone || null,
    session_id: S.session, decision: d, reason: r.reason_code, user: r.identity?.user_id || null,
    p_live: r.liveness_score != null ? Math.round(r.liveness_score * 100) / 100 : null,
    sim: r.identity?.similarity != null ? Math.round(r.identity.similarity * 100) / 100 : null,
    total_ms: meta.total != null ? Math.round(meta.total) : null, offline: S.offline,
    offline_rule: r.offline_rule || null,
    scenario_label: 'offline_sync',   // own row in the scorecard, never mixed with red-team labels
  };
  S.audit.unshift(ev); S.audit = S.audit.slice(0, 40);
  if (S.offline) { S.queue.push(ev); saveQueue(); }
  renderAudit();
}

function renderAudit() {
  const ul = $('audit');
  ul.innerHTML = S.audit.length ? S.audit.map((e) => `
    <li>
      <span class="t">${esc(new Date(e.ts).toLocaleTimeString([], { hour12: false }))}</span>
      <span class="tag ${e.decision}">${e.decision === 'STEP_UP' ? 'STEP-UP' : e.decision}</span>
      <span class="r" title="${esc(e.reason)}">${esc(e.reason)}${e.user ? ' · ' + esc(e.user) : ''}${e.offline ? '<span class="tag off">OFFLINE</span>' : ''}</span>
      <span class="t">${fmtMs(e.total_ms)}</span>
    </li>`).join('') : '<li class="muted">No decisions yet.</li>';
  $('queue').hidden = S.queue.length === 0;
  $('queue-count').textContent = S.queue.length;
  $('btn-sync').disabled = S.offline;
  $('btn-sync').textContent = S.offline ? 'Waiting for link' : 'Sync now';
}

async function syncQueue() {
  if (S.offline || !S.queue.length) return;
  const batch = S.queue.slice();
  try {
    const r = await api('/api/offline-sync', { method: 'POST', body: { events: batch } });
    const sent = new Set(batch.map((e) => e.event_id));
    S.queue = S.queue.filter((e) => !sent.has(e.event_id));   // delete local copies once confirmed
    saveQueue(); renderAudit();
    toast(`Synced ${r?.synced_events ?? batch.length} offline decisions · local copies deleted`);
    refreshStats();
  } catch (e) {
    toast('Sync failed, records kept locally: ' + e.message);
  }
}

function setOffline(on) {
  S.offline = on;
  document.body.classList.toggle('offline', on);
  $('offline-banner').hidden = !on;
  $('cam-badge').hidden = !on;
  $('offline-text').textContent = on ? 'Central link DOWN (simulated)' : 'Simulate central link loss';
  renderStatus(); renderAudit();
  if (!on) syncQueue();   // connection back → push queued decisions
}

// ------------------------------------------------------------------ scorecard
async function refreshStats() {
  let data;
  try { data = await api('/api/stats', { timeout: 4000 }); } catch { return; }
  const rows = (data.rows || []);
  const body = $('stats-body');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="9" class="muted">No attempts yet. Pick a test label, then run genuine and attack attempts.</td></tr>';
    $('summary').innerHTML = '';
    return;
  }
  const NAMES = { genuine: 'Genuine', print: 'Printed photo', screen_photo: 'Screen photo', video_replay: 'Video replay', mask: 'Mask', unlabeled: 'Unlabelled', offline_sync: 'Offline decisions (synced)' };
  const scored = (r) => r.scenario !== 'unlabeled' && r.scenario !== 'offline_sync';
  body.innerHTML = rows.map((r) => {
    const stepUp = r.n - r.allow - r.deny;              // REQUEST_VERIFICATION + policy STEP_UP
    let status = '<span class="chip">n/a</span>';
    let metric = r.key_metric;
    if (r.scenario === 'offline_sync') { status = '<span class="chip warn">synced</span>'; metric = 'decided on kiosk while offline'; }
    else if (r.scenario !== 'unlabeled') {
      if (r.kind === 'attack') status = r.allow === 0 ? '<span class="chip ok">blocked</span>' : '<span class="chip bad">leaks</span>';
      else status = r.deny_rate <= 0.05 ? '<span class="chip ok">usable</span>' : '<span class="chip warn">high rejection</span>';
    }
    return `<tr>
      <td>${esc(NAMES[r.scenario] || r.scenario)}</td><td class="num">${r.n}</td>
      <td class="num">${r.allow}</td><td class="num">${stepUp}</td><td class="num">${r.deny}</td>
      <td>${esc(metric)}</td><td class="num">${fmtMs(r.median_ms)}</td><td class="num">${fmtMs(r.p95_ms)}</td><td>${status}</td></tr>`;
  }).join('');

  const att = rows.filter((r) => r.kind === 'attack' && scored(r));
  const gen = rows.find((r) => r.scenario === 'genuine');
  const attN = att.reduce((a, r) => a + r.n, 0), attAllow = att.reduce((a, r) => a + r.allow, 0);
  const worstP95 = Math.max(...rows.map((r) => r.p95_ms || 0));
  const total = rows.reduce((a, r) => a + r.n, 0);
  // Rule of three: with 0 accepted attacks out of n, the 95% upper bound is ≈ 3/n.
  const attackStat = attN ? (attAllow === 0 ? `0/${attN} (≤ ${Math.min(100, (300 / attN)).toFixed(1)}%)` : `${attAllow}/${attN}`) : '—';
  $('summary').innerHTML = `
    <div><b>${total}</b><span>labelled + unlabelled attempts</span></div>
    <div><b>${attackStat}</b><span>attacks accepted (95% bound)</span></div>
    <div><b>${gen ? pct(gen.deny_rate) : '—'}</b><span>genuine wrongly denied</span></div>
    <div><b>${fmtMs(worstP95)}</b><span>worst p95 server time</span></div>`;
}

// ------------------------------------------------------------------ admin
async function enroll() {
  const id = $('enroll-id').value.trim().toLowerCase();
  const msg = $('enroll-msg');
  if (!/^[a-z0-9-]{2,32}$/.test(id)) { msg.className = 'small bad-text'; msg.textContent = 'Use 2–32 characters: a-z, 0-9 or "-" (no underscore).'; return; }
  if (!camReady()) { msg.className = 'small bad-text'; msg.textContent = 'Camera not ready.'; return; }
  $('btn-enroll').disabled = true;
  msg.className = 'small'; msg.textContent = 'Capturing… look at the camera';
  try {
    const { frames } = await captureBurst(ENROLL_BURST);
    msg.textContent = 'Creating template…';
    const r = await api('/api/enroll', { method: 'POST', timeout: 10000, body: { user_id: id, frames } });
    msg.className = 'small ok-text';
    msg.textContent = `Enrolled "${r.user_id}" from ${r.frames_used} good frames. Template is revocable; no photo stored.`;
    heartbeat();
  } catch (e) {
    msg.className = 'small bad-text'; msg.textContent = 'Enroll failed: ' + e.message;
  } finally { $('btn-enroll').disabled = false; }
}

async function revoke() {
  const id = $('revoke-id').value;
  const msg = $('revoke-msg');
  if (!id) return;
  if (!confirm(`Revoke the template for "${id}"? They must enroll again to get access.`)) return;
  try {
    await api('/api/users/' + encodeURIComponent(id), { method: 'DELETE' });
    msg.className = 'small ok-text'; msg.textContent = `Template and key for "${id}" deleted.`;
    heartbeat();
  } catch (e) { msg.className = 'small bad-text'; msg.textContent = 'Revoke failed: ' + e.message; }
}

// ------------------------------------------------------------------ wiring
function setTimeOverride(v) {
  S.timeOverride = v || '';
  document.querySelectorAll('.seg button').forEach((b) => {
    b.classList.toggle('on', b.dataset.time === S.timeOverride || (!S.timeOverride && b.dataset.time === ''));
    b.classList.toggle('demo', b.dataset.time !== '');
  });
  if (S.timeOverride && ![...document.querySelectorAll('.seg button')].some((b) => b.dataset.time === S.timeOverride)) {
    document.querySelectorAll('.seg button').forEach((b) => b.classList.remove('on'));
  }
  if (!S.timeOverride) $('time-custom').value = '';
  renderStatus();
}

function init() {
  $('r-live').textContent = OFFLINE.minLive; $('r-sim').textContent = OFFLINE.minSim;
  $('btn-verify').addEventListener('click', () => verify());
  $('btn-recapture').addEventListener('click', () => verify({ reuseSession: true }));
  $('btn-action').addEventListener('click', () => doStep('ACTION'));
  $('btn-otp').addEventListener('click', () => {
    const otp = $('otp').value.trim();
    if (!/^\d{6}$/.test(otp)) { $('step-err').textContent = 'Enter the 6-digit code.'; return; }
    doStep('OTP', { otp });
  });
  $('otp').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btn-otp').click(); });
  $('btn-otp-demo').addEventListener('click', async () => {
    try { const r = await api('/api/otp/' + encodeURIComponent(S.session)); $('otp-demo').textContent = r.otp ? `SMS: ${r.otp}` : 'no code issued for this session'; }
    catch (e) { $('otp-demo').textContent = 'unavailable: ' + e.message; }
  });
  $('cam-retry').addEventListener('click', startCamera);
  $('offline-toggle').addEventListener('change', (e) => setOffline(e.target.checked));
  $('btn-sync').addEventListener('click', syncQueue);
  $('btn-stats').addEventListener('click', refreshStats);
  document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => setTimeOverride(b.dataset.time)));
  $('time-custom').addEventListener('change', (e) => setTimeOverride(e.target.value));
  if ($('btn-admin')) $('btn-admin').addEventListener('click', () => { heartbeat(); $('admin').showModal(); });
  $('btn-enroll').addEventListener('click', enroll);
  $('btn-revoke').addEventListener('click', revoke);
  $('zone').addEventListener('change', () => { hideStep(); setHero('idle', 'READY', 'Step into the frame', `Zone: ${ZONES[$('zone').value]}`); });
  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (e.code === 'Space' && !['input', 'select', 'textarea', 'button'].includes(tag) && !$('admin').open) {
      e.preventDefault(); verify();
    }
  });

  setTimeOverride('');
  renderAudit();
  startCamera();
  heartbeat();
  setInterval(heartbeat, HEALTH_EVERY_MS);
  refreshStats();
}

document.addEventListener('DOMContentLoaded', init);