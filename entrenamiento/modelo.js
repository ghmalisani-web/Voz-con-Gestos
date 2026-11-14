// Gemelo en JavaScript de features() y del modelo de train.py. Se pega tal cual dentro de la app.
const M_T = 20, M_RAW = 98, M_FPF = 102;
function mFill(raw) {                       // raw: Float32Array(T*98) con NaN donde falta
  const out = Float32Array.from(raw);
  for (let j = 0; j < M_RAW; j++) {
    let first = -1;
    for (let t = 0; t < M_T; t++) if (!Number.isNaN(out[t * M_RAW + j])) { first = t; break; }
    if (first < 0) { for (let t = 0; t < M_T; t++) out[t * M_RAW + j] = 0; continue; }
    let last = out[first * M_RAW + j];
    for (let t = 0; t < M_T; t++) {
      const v = out[t * M_RAW + j];
      if (Number.isNaN(v)) out[t * M_RAW + j] = last; else last = v;
    }
  }
  return out;
}
function mFeatures(raw) {
  const r = mFill(raw), F = new Float32Array(M_T * M_FPF);
  for (let t = 0; t < M_T; t++) {
    const b = t * M_RAW, o0 = t * M_FPF;
    for (let h = 0; h < 2; h++) {
      const p = b + h * 42, o = o0 + h * 45;
      const wx = r[p], wy = r[p + 1], mx = r[p + 18] - wx, my = r[p + 19] - wy;
      const n = Math.max(Math.hypot(mx, my), 1e-6), ca = -my / n, sa = -mx / n;
      let sum = 0; for (let k = 0; k < 42; k++) sum += Math.abs(r[p + k]);
      const has = sum > 0 ? 1 : 0;
      F[o] = Number.isNaN(raw[p]) ? 0 : 1;
      F[o + 1] = wx * has; F[o + 2] = wy * has;
      for (let k = 1; k < 21; k++) {
        const x = (r[p + k * 2] - wx) / n, y = (r[p + k * 2 + 1] - wy) / n;
        F[o + 3 + (k - 1) * 2] = (x * ca - y * sa) * has;
        F[o + 4 + (k - 1) * 2] = (x * sa + y * ca) * has;
      }
      F[o + 43] = sa * has; F[o + 44] = ca * has;
    }
    for (let k = 0; k < 8; k++) F[o0 + 90 + k] = r[b + 90 + k];
  }
  const vi = [1, 2, 46, 47];
  for (let t = 0; t < M_T; t++) for (let k = 0; k < 4; k++) {
    const cur = F[t * M_FPF + vi[k]], prev = t ? F[(t - 1) * M_FPF + vi[k]] : cur;
    F[t * M_FPF + 98 + k] = (cur - prev) * 4;
  }
  return F;
}
function mDecode(b64) {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function mLoad(json) {
  return { classes: json.classes, mu: mDecode(json.mu), sd: mDecode(json.sd),
    layers: json.layers.map((l) => ({ nin: l.in, nout: l.out, W: mDecode(l.W), b: mDecode(l.b) })) };
}
function mPredict(model, raw) {             // devuelve probabilidades por clase
  let x = mFeatures(raw);
  for (let i = 0; i < x.length; i++) x[i] = (x[i] - model.mu[i]) / model.sd[i];
  model.layers.forEach((L, li) => {
    const y = new Float32Array(L.nout);
    for (let j = 0; j < L.nout; j++) y[j] = L.b[j];
    for (let i = 0; i < L.nin; i++) { const xi = x[i]; if (xi === 0) continue; const row = i * L.nout; for (let j = 0; j < L.nout; j++) y[j] += xi * L.W[row + j]; }
    if (li < model.layers.length - 1) for (let j = 0; j < L.nout; j++) if (y[j] < 0) y[j] = 0;
    x = y;
  });
  if (x.length === 1) { const p = 1 / (1 + Math.exp(-x[0])); return Float32Array.of(1 - p, p); }
  let mx = -Infinity; for (const v of x) if (v > mx) mx = v;
  let s = 0; for (let j = 0; j < x.length; j++) { x[j] = Math.exp(x[j] - mx); s += x[j]; }
  for (let j = 0; j < x.length; j++) x[j] /= s;
  return x;
}
if (typeof module !== "undefined") module.exports = { mFeatures, mLoad, mPredict };
