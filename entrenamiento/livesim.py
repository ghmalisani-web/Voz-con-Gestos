import sys, json, base64, numpy as np
sys.path.insert(0, '/home/claude/train'); import train
N=["nada","Opaco","Rojo","Verde","Amarillo","Brillante","Celeste","Colores","Rosa","Mujer","Enemigo","Hijo","Hombre","Lejos","Cajón","Nacer","Aprender"]
def load_model(p):
    m=json.load(open(p)); dec=lambda s:np.frombuffer(base64.b64decode(s),'<f4')
    W=lambda l:(np.frombuffer(base64.b64decode(l['W']),np.int8).astype(np.float32)*l['q'] if 'q' in l else dec(l['W'])).reshape(l['in'],l['out'])
    L=[(W(l),dec(l['b'])) for l in m['layers']]
    return m['classes'],dec(m['mu']),dec(m['sd']),L
def predict(model,raw):
    cl,mu,sd,L=model; x=(train.features(raw)-mu)/sd
    for i,(W,b) in enumerate(L):
        x=x@W+b
        if i<len(L)-1: x=np.maximum(x,0)
    e=np.exp(x-x.max()); return e/e.sum()
def frames_px(rec, left_slot0=True):
    out=[]
    for f in rec:
        fr=np.full(98,np.nan,np.float32); used=[False,False]
        for h in f['hands']:
            idx=(0 if h['label']=='Left' else 1) if left_slot0 else (1 if h['label']=='Left' else 0)
            if used[idx]: idx=1-idx
            if used[idx]: continue
            used[idx]=True; fr[idx*42:idx*42+42]=np.array(h['lm'],np.float32).reshape(-1)
        fr[84:98]=np.array(f['pose'],np.float32).reshape(-1); out.append(fr)
    return np.stack(out)
def window_raw(px, t_ms, now, D):
    sel=np.where((t_ms>=now-D)&(t_ms<=now))[0]
    if len(sel)<8 or now-t_ms[sel[0]]<D*0.8: return None
    hand=(~np.isnan(px[sel,0]))|(~np.isnan(px[sel,42]))
    if hand.mean()<0.5: return None
    ts=np.linspace(now-D,now,train.T); idx=[sel[np.argmin(np.abs(t_ms[sel]-x))] for x in ts]
    w=px[idx].copy(); sh=px[sel]
    cx=np.nanmedian((sh[:,86]+sh[:,88])/2); cy=np.nanmedian((sh[:,87]+sh[:,89])/2); s=np.nanmedian(np.hypot(sh[:,86]-sh[:,88],sh[:,87]-sh[:,89]))
    w[:,0::2]=(w[:,0::2]-cx)/s; w[:,1::2]=(w[:,1::2]-cy)/s
    return w
if __name__=="__main__":
    model=load_model(sys.argv[1]); rec=json.load(open('/tmp/smoke/user20.json')); t_ms=np.arange(len(rec))*50.0
    conf=float(sys.argv[2]) if len(sys.argv)>2 else 0.85
    px=frames_px(rec,False); hand=(~np.isnan(px[:,0]))|(~np.isnan(px[:,42]))
    print('mano por cuadro:',''.join('1' if h else '.' for h in hand))
    prev=None; blocked=None; fires=[]; line=[]
    for i in range(0,len(rec),3):
        now=t_ms[i]; best=None
        recent=[j for j in range(max(0,i-5),i+1) if hand[j]]
        if recent:
            for D in (800,1200,1700,2300):
                w=window_raw(px,t_ms,now,D)
                if w is None: continue
                p=predict(model,w); k=int(np.argmax(p[1:]))+1
                if best is None or p[k]>best[1]: best=(model[0][k],float(p[k]))
        line.append('%.1fs:%s %.2f'%(now/1000,N[best[0]] if best else '-',best[1] if best else 0))
        ok=best and best[1]>=conf
        if ok and best[0]==blocked: prev=None; continue
        if blocked and not ok: blocked=None
        if not ok: prev=None; continue
        if prev==best[0]: fires.append((round(now/1000,2),N[best[0]])); blocked=best[0]; prev=None
        else: prev=best[0]
    print(' | '.join(line)); print('DISPAROS (umbral %.2f):'%conf, fires)
