"""Entrena el modelo de reconocimiento de señas de Voz con Gestos a partir de los archivos
de datos que exporta la app (posiciones de manos y cuerpo de los videos de LSA64).

Uso:  python3 train.py datos1.json [datos2.json ...] [--out modelo.json] [--quick]
"""
import sys, json, base64, argparse
import numpy as np
from sklearn.neural_network import MLPClassifier

T = 20            # cuadros por secuencia
RAW = 98          # mano0[21xy] mano1[21xy] pose[7xy]
FPF = 102         # rasgos por cuadro
NADA = 0          # clase "ninguna seña"


# ---------- carga ----------
def load(paths):
    items, seen = [], set()
    for p in paths:
        d = json.load(open(p))
        assert d["fmt"] == "voz-con-gestos/lsa64-landmarks" and d["frames"] == T and d["dim"] == RAW, p
        for it in d["items"]:
            if it["f"] in seen:
                continue
            seen.add(it["f"])
            a = np.frombuffer(base64.b64decode(it["d"]), dtype="<i2").reshape(T, RAW).astype(np.float32)
            miss = a == d["miss"]
            a = a / d["scale"]
            a[miss] = np.nan
            items.append((it["s"], it["p"], it["r"], a))
    y = np.array([i[0] for i in items]); subj = np.array([i[1] for i in items])
    X = np.stack([i[3] for i in items])
    return X, y, subj


# ---------- rasgos (esta función tiene su gemela exacta en JavaScript dentro de la app) ----------
def fill(seq):
    """Rellena cuadros sin dato con el cuadro válido más cercano (hacia atrás y luego hacia adelante)."""
    out = seq.copy()
    for j in range(seq.shape[1]):
        col = out[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            col[:] = 0.0
            continue
        idx = np.where(ok, np.arange(len(col)), -1)
        idx = np.maximum.accumulate(idx)
        first = np.argmax(ok)
        idx[idx < 0] = first
        col[:] = col[idx]
    return out


def features(raw):
    """raw: (T, 98) en unidades de hombros, NaN = no detectado. Devuelve (T*FPF,)."""
    present = np.stack([~np.isnan(raw[:, 0]), ~np.isnan(raw[:, 42])], 1).astype(np.float32)   # (T,2)
    r = fill(raw)
    F = np.zeros((T, FPF), np.float32)
    for h in range(2):
        pts = r[:, h * 42:(h + 1) * 42].reshape(T, 21, 2)
        w = pts[:, 0]                                  # muñeca
        m = pts[:, 9] - w                              # muñeca -> nudillo medio
        n = np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-6)
        ca, sa = -m[:, 1] / n, -m[:, 0] / n            # rotación que deja ese vector hacia arriba
        rel = (pts[:, 1:] - w[:, None]) / n[:, None, None]
        x = rel[..., 0] * ca[:, None] - rel[..., 1] * sa[:, None]
        yv = rel[..., 0] * sa[:, None] + rel[..., 1] * ca[:, None]
        has = (np.abs(pts).sum((1, 2)) > 0).astype(np.float32)   # la mano tiene datos (propios o rellenados)
        o = h * 45
        F[:, o] = present[:, h]
        F[:, o + 1:o + 3] = w * has[:, None]
        F[:, o + 3:o + 43] = np.stack([x, yv], -1).reshape(T, 40) * has[:, None]
        F[:, o + 43] = sa * has
        F[:, o + 44] = ca * has
    F[:, 90:98] = r[:, 84 + 6:84 + 14]                 # codos y muñecas del cuerpo
    v = np.diff(np.concatenate([F[:1, [1, 2, 46, 47]], F[:, [1, 2, 46, 47]]]), axis=0)
    F[:, 98:102] = v * 4                               # velocidad de las muñecas
    return F.reshape(-1)


# ---------- aumentos ----------
def resample(raw, t0, t1, n=T):
    """Toma el tramo [t0,t1] (en fracción del clip) y lo lleva a n cuadros (cuadro más cercano)."""
    idx = np.clip(np.round(np.linspace(t0, t1, n) * (T - 1)).astype(int), 0, T - 1)
    return raw[idx]


def embed(raw, rng, f):
    """Simula el uso en vivo: la seña ocupa una fracción f de la ventana; antes la mano sube desde abajo
    (o todavía no se ve) y después baja (o desaparece). Los videos de LSA64 vienen recortados y no traen eso."""
    n = max(2, int(round(f * T))); rest = T - n
    pre = int(rng.integers(0, rest + 1)); post = rest - pre
    mid = resample(raw, 0, 1, n)
    def ramp(frame, k, up):
        if k == 0: return np.zeros((0, RAW), np.float32)
        out = np.repeat(frame[None], k, 0).copy()
        if rng.random() < 0.35:
            out[:, 0:84] = np.nan                         # la mano todavía no se ve / ya no se ve
        else:
            d = rng.uniform(0.9, 1.6)
            off = np.linspace(d, d / (k + 1), k) if up else np.linspace(d / (k + 1), d, k)
            out[:, 1:84:2] += off[:, None]
            if rng.random() < 0.5: out[: k // 2 if up else 0, 0:84] = np.nan
        return out
    return np.concatenate([ramp(mid[0], pre, True), mid, ramp(mid[-1], post, False)]).astype(np.float32)


def mirror(raw):
    out = raw.copy()
    out[:, 0:42], out[:, 42:84] = raw[:, 42:84], raw[:, 0:42]
    out[:, 0:84:2] *= -1
    pose = raw[:, 84:98].reshape(T, 7, 2).copy()
    pose = pose[:, [0, 2, 1, 4, 3, 6, 5]]
    pose[..., 0] *= -1
    out[:, 84:98] = pose.reshape(T, 14)
    return out


def augment(raw, rng, live=True):
    a = resample(raw, rng.uniform(0, 0.15), rng.uniform(0.85, 1.0))
    if live and rng.random() < 0.5:
        a = embed(a, rng, rng.uniform(0.6, 0.95))
    if rng.random() < 0.3:                           # sostener el final (la persona se queda quieta)
        k = rng.integers(2, 6)
        a = np.concatenate([resample(a, 0, 1)[::1][: T - k], np.repeat(a[-1:], k, 0)])
    s = rng.uniform(0.9, 1.1)
    a = a * s
    a[:, 0:84:2] += rng.normal(0, 0.06); a[:, 1:84:2] += rng.normal(0, 0.06)     # mano corrida respecto del cuerpo
    a = a + rng.normal(0, 0.006, a.shape).astype(np.float32)
    if rng.random() < 0.25:                          # cuadros perdidos (mano delante de la cara)
        i = rng.integers(0, T - 3); a[i:i + rng.integers(1, 4), 0:84] = np.nan
    if rng.random() < 0.25:                          # la otra mano no se ve
        h = rng.integers(0, 2)
        if np.isnan(a[:, (1 - h) * 42]).mean() < 0.5:
            a[:, h * 42:(h + 1) * 42] = np.nan
    return a


def negatives(X, y, rng, n):
    """Ejemplos de 'ninguna seña': sin manos, mano abajo, mano subiendo o bajando, mezcla de dos señas."""
    out = []
    for _ in range(n):
        k = rng.integers(0, 8)
        i = rng.integers(len(X))
        a = X[i].copy()
        if k >= 5:                                        # apenas un pedacito de seña entre subir y bajar la mano
            out.append(augment(embed(a, rng, rng.uniform(0.05, 0.3)), rng, live=False)); continue
        if k == 0:
            a[:, 0:84] = np.nan
        elif k == 1:
            a[:, 1:84:2] += rng.uniform(0.9, 1.6)
            a = resample(a, *sorted(rng.uniform(0, 1, 2)))
        elif k in (2, 3):
            ramp = np.linspace(rng.uniform(1.0, 1.6), 0, T)[:, None] if k == 2 else np.linspace(0, rng.uniform(1.0, 1.6), T)[:, None]
            f = a[0:1] if k == 2 else a[-1:]
            a = np.repeat(f, T, 0); a[:, 1:84:2] += ramp
        else:
            j = rng.integers(len(X))
            if y[j] == y[i]:
                j = (j + 1) % len(X)
            b = X[j]
            a = np.concatenate([resample(a, 0.5, 1)[: T // 2], resample(b, 0, 0.5)[: T - T // 2]])
        out.append(augment(a, rng, live=False) if k != 0 else a)
    return out


def build(X, y, rng, n_aug, mirror_too=True, neg_frac=0.25):
    Xs, ys = [], []
    for raw, lab in zip(X, y):
        Xs.append(features(raw)); ys.append(lab)
        for _ in range(n_aug):
            Xs.append(features(augment(raw, rng))); ys.append(lab)
        if mirror_too:
            mraw = mirror(raw)
            Xs.append(features(mraw)); ys.append(lab)
            for _ in range(max(1, n_aug // 2)):
                Xs.append(features(augment(mraw, rng))); ys.append(lab)
    for a in negatives(X, y, rng, int(len(Xs) * neg_frac)):
        Xs.append(features(a)); ys.append(NADA)
    return np.stack(Xs), np.array(ys)


def fit(Xtr, ytr, quick, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
                        max_iter=25 if quick else 80, early_stopping=False, random_state=seed)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="modelo.json")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--aug", type=int, default=4)
    ap.add_argument("--min", type=int, default=40, help="descartar señas con menos videos que esto")
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    X, y, subj = load(a.files)
    rng = np.random.default_rng(0)
    keep = np.isin(y, [c for c in set(y) if (y == c).sum() >= a.min])
    X, y, subj = X[keep], y[keep], subj[keep]
    classes = sorted(set(y))
    print(f"{len(X)} videos · {len(classes)} señas · personas {sorted(set(subj))}")

    # --- validación: se entrena sin una persona y se prueba con ella ---
    if not a.no_cv and len(classes) >= 2 and len(set(subj)) >= 3:
        hit = tot = 0; per_subj = {}; conf = {}
        for s in sorted(set(subj)):
            tr, te = subj != s, subj == s
            Xtr, ytr = build(X[tr], y[tr], rng, a.aug)
            clf, mu, sd = fit(Xtr, ytr, a.quick)
            Xte = np.stack([features(r) for r in X[te]])
            pred = clf.predict((Xte - mu) / sd)
            ok = (pred == y[te]); hit += ok.sum(); tot += len(ok); per_subj[int(s)] = round(float(ok.mean()), 3)
            for t, p in zip(y[te], pred):
                if t != p: conf[(int(t), int(p))] = conf.get((int(t), int(p)), 0) + 1
        print(f"Aciertos con personas no vistas: {hit}/{tot} = {hit / tot:.1%}")
        print("Por persona:", per_subj)
        print("Confusiones más frecuentes (real, dicho):", sorted(conf.items(), key=lambda kv: -kv[1])[:15])

    # --- modelo final con todos los datos ---
    Xtr, ytr = build(X, y, rng, a.aug)
    clf, mu, sd = fit(Xtr, ytr, a.quick)
    b64 = lambda arr: base64.b64encode(np.asarray(arr, "<f4").tobytes()).decode()
    model = {"fmt": "voz-con-gestos/modelo", "v": 1, "T": T, "fpf": FPF, "classes": [int(c) for c in clf.classes_],
             "mu": b64(mu), "sd": b64(sd),
             "layers": []}
    for W, b in zip(clf.coefs_, clf.intercepts_):          # pesos en 8 bits (4 veces más liviano)
        q = float(np.abs(W).max()) / 127
        model["layers"].append({"in": int(W.shape[0]), "out": int(W.shape[1]), "q": q,
                                "W": base64.b64encode(np.clip(np.round(W / q), -127, 127).astype(np.int8).tobytes()).decode(), "b": b64(b)})
    json.dump(model, open(a.out, "w"))
    print("Modelo guardado en", a.out, f"({len(json.dumps(model)) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
