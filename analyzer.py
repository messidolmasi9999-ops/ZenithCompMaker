"""ZenithCompMaker analiz + render motoru.
Hat: ffmpeg(NVDEC) -> YOLO(oyuncu+top, fp16/Tensor) -> takip -> yüz(ArcFace)+forma no/isim(OCR) kanıtı
-> kareler arası kimlik birleştirme -> aksiyon çıkarımı -> NVENC ile paralel kesim -> tek MP4."""
import os, re, json, time, queue, difflib, threading, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2

class Cancelled(Exception): pass

MODES = {"hizli": (3, 1280), "dengeli": (5, 1280), "detayli": (8, 1600), "maksimum": (12, 1920)}
MAX_COMP = 20 * 60
W_EV = {"shot": 5, "dribble": 3, "pass": 2, "touch": 1}
PRE_POST = {"shot": (4, 6), "dribble": (2, 2), "pass": (3, 3), "touch": (2, 3)}
_models = {}

def run(cmd): return subprocess.run(cmd, capture_output=True, text=True)

def norm(s):
    return re.sub(r"[^A-Z]", "", s.upper().translate(str.maketrans("İıŞşĞğÜüÖöÇç", "IISSGGUUOOCC")))

def probe(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate:format=duration", "-of", "json", path])
    j = json.loads(r.stdout or "{}"); s = (j.get("streams") or [{}])[0]
    try:
        a, b = s.get("r_frame_rate", "25/1").split("/"); fps = float(a) / (float(b) or 1) or 25.0
    except Exception: fps = 25.0
    return dict(w=int(s.get("width", 1280)), h=int(s.get("height", 720)), fps=fps,
                dur=float((j.get("format") or {}).get("duration") or 0))

def list_devices():
    out = [{"id": "cpu", "label": "CPU"}]
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            out.insert(i, {"id": f"cuda:{i}", "label": f"GPU: {torch.cuda.get_device_name(i)} (CUDA + Tensor)"})
    except Exception: pass
    return out

_dets = {}

def load_models(dev, need_ocr, wname="yolov8m"):
    m = _load_base(dev, need_ocr)
    k = (dev, wname if m["gpu"] else "yolov8n")
    if k not in _dets:
        from ultralytics import YOLO; _dets[k] = YOLO(k[1] + ".pt")
    m["det"] = _dets[k]; return m

def _load_base(dev, need_ocr):
    key = (dev, need_ocr)
    if key in _models: return _models[key]
    import torch
    from insightface.app import FaceAnalysis
    try:
        import onnxruntime as ort; ort.preload_dlls()
    except Exception: pass
    gpu = dev.startswith("cuda")
    if gpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"])
    ocr = None
    if need_ocr:
        import easyocr; ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    m = dict(fa=fa, ocr=ocr, gpu=gpu, ctx=int(dev.split(":")[1]) if gpu else -1)
    _models[key] = m; return m

def ref_embeddings(m, paths):
    m["fa"].prepare(ctx_id=m["ctx"], det_size=(640, 640)); embs = []
    for p in paths:
        img = cv2.imread(p); fs = m["fa"].get(img) if img is not None else []
        if fs: embs.append(max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding)
    if not embs: raise RuntimeError("Fotoğraflarda yüz bulunamadı. Net ve önden çekilmiş yüz fotoğrafı yükle.")
    m["fa"].prepare(ctx_id=m["ctx"], det_size=(320, 320))
    return np.array(embs)

def iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0

class Track:
    def __init__(s, id, box, i): s.id, s.box, s.last, s.n, s.score, s.vel = id, box, i, 1, 0.0, np.zeros(2)

class Tracker:
    def __init__(s, max_age): s.tracks, s.nid, s.max_age = {}, 1, max_age
    @staticmethod
    def _pred(t, i):
        dx, dy = t.vel * (i - t.last)
        return (t.box[0] + dx, t.box[1] + dy, t.box[2] + dx, t.box[3] + dy)
    def update(s, i, boxes):
        alive = [t for t in s.tracks.values() if i - t.last <= s.max_age]
        pairs = sorted(((iou(s._pred(t, i), b), ti, bi) for ti, t in enumerate(alive) for bi, b in enumerate(boxes)), reverse=True)
        ut, ub, out = set(), set(), {}
        for v, ti, bi in pairs:
            if v < 0.2: break
            if ti in ut or bi in ub: continue
            ut.add(ti); ub.add(bi); t = alive[ti]; d = max(1, i - t.last); b = boxes[bi]
            t.vel = 0.5 * t.vel + 0.5 * np.array([((b[0] + b[2]) - (t.box[0] + t.box[2])) / 2 / d, ((b[1] + b[3]) - (t.box[1] + t.box[3])) / 2 / d])
            t.box, t.last, t.n = b, i, t.n + 1; out[bi] = t
        for bi, b in enumerate(boxes):
            if bi not in ub:
                t = Track(s.nid, b, i); s.nid += 1; s.tracks[t.id] = t; out[bi] = t
        return out

def extract(rec, tids):
    """Hedef oyuncunun kayıtlarından olayları çıkarır: touch/pass/shot/dribble/run."""
    pts = []
    for t, pl, ball in rec:
        pts.append((t, next((b for tid, b in pl if tid in tids), None), ball))
    li = None
    for i, p in enumerate(pts):
        if p[2] is not None:
            if li is not None and 1 < i - li <= 3:
                a0, b0 = pts[li][2], p[2]
                for k in range(li + 1, i):
                    fr_ = (k - li) / (i - li); pts[k] = (pts[k][0], pts[k][1], tuple(a0[z] + (b0[z] - a0[z]) * fr_ for z in range(4)))
            li = i
    n = len(pts); contact = [False] * n; seen = 0
    ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    for i, (t, bx, ball) in enumerate(pts):
        if bx is None: continue
        seen += 1
        if ball is not None:
            bh = bx[3] - bx[1]; bc = ctr(ball)
            contact[i] = ((bc[0] - (bx[0] + bx[2]) / 2) ** 2 + (bc[1] - (bx[1] + 0.85 * bh)) ** 2) ** 0.5 < 0.55 * bh
    ev, i = [], 0
    while i < n:
        if not contact[i]: i += 1; continue
        j = i
        while (j + 1 < n and contact[j + 1]) or (j + 2 < n and contact[j + 2]): j += 1
        ts, te = pts[i][0], pts[j][0]; bi = next(k for k in range(i, j + 1) if pts[k][1]); bj = next(k for k in range(j, i - 1, -1) if pts[k][1])
        b0, b1 = pts[bi][1], pts[bj][1]; bh = max(1, b1[3] - b1[1])
        mv = ((ctr(b1)[0] - ctr(b0)[0]) ** 2 + (ctr(b1)[1] - ctr(b0)[1]) ** 2) ** 0.5 / bh
        kind = None
        if te - ts >= 1.2 and mv > 1.0: ev.append((ts, te, "dribble")); kind = "d"
        sp = 0
        for k in range(j + 1, min(n, j + 4)):
            if pts[k][2] is not None and pts[j][2] is not None and pts[k][0] > te:
                sp = ((ctr(pts[k][2])[0] - ctr(pts[j][2])[0]) ** 2 + (ctr(pts[k][2])[1] - ctr(pts[j][2])[1]) ** 2) ** 0.5 / bh / (pts[k][0] - te + 1e-6); break
        if sp > 7: ev.append((te, te, "shot"))
        elif sp > 3: ev.append((te, te, "pass"))
        elif not kind: ev.append((ts, te, "touch"))
        i = j + 1
    run_t, last = 0.0, None
    for t, bx, _ in pts:
        if bx is None: last = None; continue
        if last and t - last[0] < 1.5:
            sp = ((ctr(bx)[0] - ctr(last[1])[0]) ** 2 + (ctr(bx)[1] - ctr(last[1])[1]) ** 2) ** 0.5 / max(1, bx[3] - bx[1]) / (t - last[0])
            if sp > 2.0: run_t += t - last[0]
        last = (t, bx)
    cnt = {k: sum(1 for e in ev if e[2] == k) for k in ("touch", "pass", "shot", "dribble")}
    cnt["run_sec"] = round(run_t); cnt["seen_sec"] = round(seen * (pts[1][0] - pts[0][0]) if n > 1 else 0)
    return ev, cnt

def make_segments(ev, dur):
    w = sorted((max(0, s - PRE_POST[k][0]), min(dur, e + PRE_POST[k][1]), W_EV[k]) for s, e, k in ev)
    merged = []
    for s, e, sc in w:
        if merged and s - merged[-1][1] <= 1.5: merged[-1][1] = max(merged[-1][1], e); merged[-1][2] += sc
        else: merged.append([s, e, sc])
    merged = [m for m in merged if m[1] - m[0] >= 3]
    if sum(m[1] - m[0] for m in merged) > MAX_COMP:
        keep, tot = [], 0
        for m in sorted(merged, key=lambda m: m[2] / (m[1] - m[0]), reverse=True):
            if tot + m[1] - m[0] <= MAX_COMP: keep.append(m); tot += m[1] - m[0]
        merged = sorted(keep)
    return [(m[0], m[1]) for m in merged]

def _put(q, x, ev):
    while not ev.is_set():
        try: q.put(x, timeout=0.5); return
        except queue.Full: pass

def _reader(job, info, W, H, fps_s, gpu, q, stop):
    def spawn(g):
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt"]
        cmd += ["-hwaccel", "cuda"] if g else []
        cmd += ["-i", job["video"], "-map", "0:v:0", "-an", "-sn", "-vf", f"fps={fps_s},scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4); job["procs"].append(p); return p
    for g in ([True, False] if gpu else [False]):
        p, got = spawn(g), 0
        while not stop.is_set():
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3: break
            _put(q, np.frombuffer(buf, np.uint8).reshape(H, W, 3), stop); got += 1
        p.kill()
        if got or stop.is_set(): break
    _put(q, None, stop)

def analyze(job):
    c, cancel = job["cfg"], job["cancel"]
    info = probe(job["video"])
    if info["dur"] <= 0: raise RuntimeError("Video okunamadı. Dosya bozuk olabilir; MP4 ile tekrar dene.")
    fps_s, W = MODES[c["mode"]]; W = min(W, info["w"]); H = int(round(info["h"] * W / info["w"] / 2) * 2)
    dev = c["device"]
    if dev == "auto": dev = list_devices()[0]["id"]
    job["phase"] = "Yapay zeka modelleri yükleniyor"
    name, number = norm(c["name"]), re.sub(r"\D", "", c["number"])
    m = load_models(dev, bool(name or number), {"hizli": "yolov8m", "dengeli": "yolov8m", "detayli": "yolov8l", "maksimum": "yolov8x"}[c["mode"]]); gpu = m["gpu"]
    if not gpu:
        import torch; torch.set_num_threads(os.cpu_count() or 4)
    refs = ref_embeddings(m, job["photos"])
    total = max(1, int(info["dur"] * fps_s)); job["frames_total"] = int(info["dur"] * info["fps"]); job["sampled_total"] = total
    q, stop = queue.Queue(maxsize=48), threading.Event()
    threading.Thread(target=_reader, args=(job, info, W, H, fps_s, gpu, q, stop), daemon=True).start()
    tracker, rec, idx, t0, last_live, last_tgt = Tracker(int(fps_s * 2)), [], 0, time.time(), 0, None
    bs, done, batch = (8 if gpu else 2), False, []
    job["phase"] = "Maç taranıyor"
    try:
        while not done:
            if cancel.is_set(): raise Cancelled()
            try: f = q.get(timeout=1)
            except queue.Empty: continue
            if f is None: done = True
            else: batch.append(f)
            if not batch or not (len(batch) >= bs or done): continue
            res = m["det"].predict(batch, imgsz=W if gpu else min(W, 960), conf=0.15, classes=[0, 32], half=gpu, device=dev if gpu else "cpu", verbose=False)
            for fr, r in zip(batch, res):
                b, cl, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
                pb = [b[k] for k in range(len(b)) if cl[k] == 0 and cf[k] >= 0.3 and (b[k][3] - b[k][1]) >= H * 0.05]
                bl = [(cf[k], b[k]) for k in range(len(b)) if cl[k] == 32]
                ball = tuple(max(bl, key=lambda x: x[0])[1]) if bl else None
                lineup = sum(1 for x in pb if (x[3] - x[1]) > 0.45 * H) >= 8   # ilk 11 grafiği: analiz etme
                tr = tracker.update(idx, pb)
                tgt_vis = any(t.score >= 3 and t.n > 1 for t in tr.values())
                if last_tgt and not tgt_vis:
                    for bi, t in tr.items():
                        if t.n == 1 and 0 < idx - last_tgt[0] <= fps_s * 2:
                            cx, cy = (pb[bi][0] + pb[bi][2]) / 2, (pb[bi][1] + pb[bi][3]) / 2
                            if ((cx - last_tgt[1]) ** 2 + (cy - last_tgt[2]) ** 2) ** 0.5 < 0.12 * W: t.score = 1.5
                nf = no = 0
                if not lineup:
                    for bi in sorted(tr, key=lambda k: -(pb[k][3] - pb[k][1])):
                        t, x = tr[bi], pb[bi]; h = x[3] - x[1]; x1, y1, x2 = max(0, int(x[0])), max(0, int(x[1])), min(W, int(x[2]))
                        if t.score >= 6 or t.score <= -4: continue
                        if h >= 100 and nf < (12 if gpu else 4) and t.n % 2 == 1:        # YÜZ: ana kimlik kanıtı
                            nf += 1; crop = fr[y1:int(y1 + 0.45 * h), x1:x2]
                            if crop.size:
                                s = max(1, 160 / max(1, min(crop.shape[:2]))); crop = cv2.resize(crop, None, fx=min(s, 3), fy=min(s, 3), interpolation=cv2.INTER_CUBIC)
                                fs = m["fa"].get(crop)
                                if fs:
                                    f0 = max(fs, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
                                    sim = float(np.max(refs @ f0.normed_embedding))
                                    t.score += 3 if sim >= 0.5 else 1.5 if sim >= 0.38 else (-1 if sim < 0.2 and f0.det_score > 0.6 else 0)
                        if m["ocr"] and h >= 120 and no < (4 if gpu else 2) and t.n % 3 == 1:   # FORMA NO / İSİM: destek kanıtı (sadece oyuncu kutusu içi)
                            no += 1; tc = fr[int(y1 + 0.12 * h):int(y1 + 0.6 * h), x1:x2]
                            if tc.size:
                                tc = cv2.resize(tc, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                for bb, txt, cfd in m["ocr"].readtext(tc):
                                    txt = txt.strip().upper().replace(" ", ""); bh = bb[2][1] - bb[0][1]
                                    if txt.isdigit() and len(txt) <= 2 and cfd >= 0.55 and bh >= 0.12 * tc.shape[0] and number:
                                        t.score += 2 if txt == number else -1.5
                                    elif name and len(norm(txt)) >= 3 and cfd >= 0.4:
                                        last_name = name if " " not in c["name"].strip() else norm(c["name"].split()[-1])
                                        if max(difflib.SequenceMatcher(None, norm(txt), z).ratio() for z in (name, last_name)) >= 0.8: t.score += 2
                        t.score = max(-6, min(6, t.score))
                tg = [t for t in tr.values() if t.score >= 3]
                if len(tg) > 1:
                    best = max(tg, key=lambda t: t.score)
                    for t in tg:
                        if t is not best: t.score = min(t.score, 2)
                for bi, t in tr.items():
                    if t.score >= 3: last_tgt = (idx, (pb[bi][0] + pb[bi][2]) / 2, (pb[bi][1] + pb[bi][3]) / 2)
                rec.append((idx / fps_s, [(tr[k].id, tuple(pb[k])) for k in tr], ball)); idx += 1
            batch = []
            el = time.time() - t0; job["sampled_done"] = idx; job["speed"] = round(idx / el, 1) if el else 0
            job["eta"] = int((total - idx) / (idx / el)) if idx and el else None
            job["progress"] = min(85, int(85 * idx / total))
            if time.time() - last_live > 2 or done:
                tids = {t.id for t in tracker.tracks.values() if t.score >= 3}
                job["live"] = extract(rec, tids)[1] if tids else {}; last_live = time.time()
    finally:
        stop.set()
        for p in job["procs"]: 
            try: p.kill()
            except Exception: pass
    tids = {t.id for t in tracker.tracks.values() if t.score >= 3}
    ev, cnt = extract(rec, tids) if tids else ([], {})
    job["live"] = cnt
    return make_segments(ev, info["dur"]), info

def _encoder():
    r = run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"])
    return "nvenc" if r.returncode == 0 else "x264"

def playable(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt:format=duration", "-of", "json", path])
    try: j = json.loads(r.stdout); d = float(j["format"]["duration"]); s = j["streams"][0]
    except Exception: return False
    if s["codec_name"] != "h264" or s["pix_fmt"] != "yuv420p" or d <= 0: return False
    return run(["ffmpeg", "-v", "error", "-ss", f"{max(0, d - 3):.2f}", "-i", path, "-t", "2", "-f", "null", "-"]).stderr.strip() == ""

def render(job, segs, info, out):
    tmp = job["dir"] / "seg"; tmp.mkdir(exist_ok=True)
    enc = _encoder(); fps = int(min(60, round(info["fps"])) or 25)
    VE = {"nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
          "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]}
    def one(k, s, e):
        for kind in dict.fromkeys([enc, "x264"]):
            if job["cancel"].is_set(): raise Cancelled()
            o = tmp / f"{k:04d}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-err_detect", "ignore_err"] + (["-hwaccel", "cuda"] if kind == "nvenc" else [])
            cmd += ["-ss", f"{s:.3f}", "-i", job["video"], "-t", f"{e - s:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", "scale=-2:'min(1080,ih)',format=yuv420p", "-r", str(fps)] + VE[kind] + ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(o)]
            p = subprocess.Popen(cmd, stderr=subprocess.DEVNULL); job["procs"].append(p); p.wait()
            if p.returncode == 0 and o.exists(): return o
        raise RuntimeError("Klip kodlanamadı")
    job["phase"] = f"Comp hazırlanıyor ({'NVENC GPU' if enc == 'nvenc' else 'CPU'} ile {len(segs)} sahne)"
    with ThreadPoolExecutor(3 if enc == "nvenc" else 2) as ex:
        futs = [ex.submit(one, k, s, e) for k, (s, e) in enumerate(segs)]
        files = []
        for k, f in enumerate(futs): files.append(f.result()); job["progress"] = 85 + int(14 * (k + 1) / len(futs))
    lst = tmp / "list.txt"; lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in files))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    if not out.exists() or not playable(out):   # garanti: oynatılamayan çıktı verme
        job["phase"] = "Çıktı doğrulanıyor / yeniden kodlanıyor"
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)])
        if not out.exists() or not playable(out): raise RuntimeError("Comp videosu doğrulanamadı.")

def process(job, out):
    segs, info = analyze(job)
    if not segs: raise RuntimeError("Oyuncu bulunamadı. Daha net yüz fotoğrafı, forma numarası/isim ekle veya 'Detaylı' modu seç.")
    job["status"] = "rendering"; render(job, segs, info, out)
    job["comp_seconds"] = round(sum(e - s for s, e in segs)); job["segments"] = len(segs)
    shutil.rmtree(job["dir"] / "seg", ignore_errors=True)
