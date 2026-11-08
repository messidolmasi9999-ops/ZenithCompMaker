import os, re, uuid, time, hmac, shutil, sqlite3, secrets, threading, subprocess, ipaddress, socket
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import analyzer

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"; UP = DATA / "uploads"; OUT = DATA / "outputs"
for p in (UP, OUT): p.mkdir(parents=True, exist_ok=True)
DB = DATA / "zenith.db"
ADMIN_PASSWORD = os.environ.get("ZENITH_ADMIN_PASSWORD", "newoxnyrozenithmaker123")
MAX_BYTES, CH = 15 * 1024 ** 3, 8 * 1024 * 1024
EXTS = {".mp4", ".mkv", ".ts", ".m2ts", ".mov", ".avi", ".webm"}
DUR = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "unlimited": None}
sk = DATA / "secret.key"
if not sk.exists(): sk.write_text(secrets.token_hex(32))
app = Flask(__name__)
app.secret_key = sk.read_text()
app.config.update(MAX_CONTENT_LENGTH=64 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=90))
jobs, jobs_lock, sem = {}, threading.Lock(), threading.Semaphore(int(os.environ.get("ZENITH_PARALLEL_JOBS", "1")))
fails = {}

def db():
    c = sqlite3.connect(DB, timeout=30); c.row_factory = sqlite3.Row; c.execute("pragma journal_mode=wal"); return c

def init_db():
    c = db()
    c.execute("create table if not exists users(id integer primary key autoincrement, username text unique not null, pw text not null, max_devices int default 1, expires text, active int default 1)")
    c.execute("create table if not exists devices(username text, token text, first_seen text, unique(username, token))")
    if not c.execute("select 1 from users").fetchone():
        c.execute("insert into users(username,pw,max_devices,expires) values(?,?,?,NULL)", ("newrasxnyroaepmaker", generate_password_hash("ZenithNNProduction"), 2))
    c.commit(); c.close()
init_db()

def expired(r): return bool(r["expires"]) and datetime.now() > datetime.strptime(r["expires"], "%Y-%m-%d %H:%M:%S")
def locked(ip): f = fails.get(ip); return bool(f and f[0] >= 8 and time.time() - f[1] < 300)
def fail(ip): f = fails.get(ip, (0, 0)); fails[ip] = (f[0] + 1, time.time())

def current_user():
    u, tok = session.get("user"), request.cookies.get("zdev")
    if not u or not tok: return None
    c = db(); r = c.execute("select * from users where username=?", (u,)).fetchone()
    ok = r and r["active"] and not expired(r) and c.execute("select 1 from devices where username=? and token=?", (u, tok)).fetchone()
    c.close(); return r["username"] if ok else None

def need_user(fn):
    @wraps(fn)
    def w(*a, **k):
        u = current_user()
        if not u: return jsonify(error="Oturum geçersiz veya hesap iptal edildi."), 401
        return fn(u, *a, **k)
    return w

def cancel_user_jobs(username):
    for j in jobs.values():
        if j["user"] == username: j["cancel"].set(); [p.kill() for p in j["procs"] if p.poll() is None]

# ---------- sayfalar ----------
@app.route("/")
def index():
    u = current_user()
    if u: return render_template("index.html", view="app", user=u)
    return render_template("index.html", view="login", error=session.pop("err", None))

@app.route("/login", methods=["POST"])
def login():
    ip = request.remote_addr
    if locked(ip): session["err"] = "Çok fazla hatalı deneme. 5 dk bekle."; return redirect("/")
    uname, pw = request.form.get("username", "").strip(), request.form.get("password", "")
    c = db(); r = c.execute("select * from users where username=?", (uname,)).fetchone()
    if not r or not check_password_hash(r["pw"], pw):
        fail(ip); c.close(); session["err"] = "Kullanıcı adı veya şifre hatalı. Bedava giriş yoktur."; return redirect("/")
    if not r["active"]: c.close(); session["err"] = "Hesabın iptal edilmiş."; return redirect("/")
    if expired(r): c.close(); session["err"] = "Üyelik süren dolmuş."; return redirect("/")
    tok = request.cookies.get("zdev") or secrets.token_urlsafe(24)
    if not c.execute("select 1 from devices where username=? and token=?", (uname, tok)).fetchone():
        if c.execute("select count(*) from devices where username=?", (uname,)).fetchone()[0] >= r["max_devices"]:
            c.close(); session["err"] = f"Cihaz limiti dolu ({r['max_devices']}/{r['max_devices']}). Yeni cihaz girişi yapılamaz."; return redirect("/")
        c.execute("insert into devices values(?,?,?)", (uname, tok, datetime.now().isoformat())); c.commit()
    c.close(); session.clear(); session.permanent = True; session["user"] = uname; fails.pop(ip, None)
    resp = redirect("/"); resp.set_cookie("zdev", tok, max_age=10 * 365 * 86400, httponly=True, samesite="Lax"); return resp

@app.route("/logout")
def logout(): session.clear(); return redirect("/")

# ---------- admin ----------
@app.route("/admin/login", methods=["POST"])
def admin_login():
    ip = request.remote_addr
    if locked(ip): session["err"] = "Çok fazla hatalı deneme. 5 dk bekle."; return redirect("/")
    if hmac.compare_digest(request.form.get("admin_password", ""), ADMIN_PASSWORD): session["admin"] = True; return redirect("/admin")
    fail(ip); session["err"] = "Admin şifresi hatalı."; return redirect("/")

@app.route("/admin")
def admin():
    if not session.get("admin"): return redirect("/")
    c = db(); rows = c.execute("select u.*, (select count(*) from devices d where d.username=u.username) used from users u order by id desc").fetchall(); c.close()
    return render_template("index.html", view="admin", users=rows, expired=expired)

@app.route("/admin/action", methods=["POST"])
def admin_action():
    if not session.get("admin"): return redirect("/")
    f, c = request.form, db(); a, uid = f.get("action"), f.get("user_id")
    if a == "create":
        d = DUR.get(f.get("duration"), None); exp = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d %H:%M:%S") if d else None
        try:
            c.execute("insert into users(username,pw,max_devices,expires) values(?,?,?,?)",
                      (f["username"].strip(), generate_password_hash(f["password"]), max(1, int(f.get("devices", 1))), exp))
        except (sqlite3.IntegrityError, ValueError, KeyError): pass
    else:
        r = c.execute("select username, active from users where id=?", (uid,)).fetchone()
        if r:
            if a == "toggle":
                c.execute("update users set active=? where id=?", (0 if r["active"] else 1, uid))
                if r["active"]: cancel_user_jobs(r["username"])
            elif a == "delete":
                c.execute("delete from users where id=?", (uid,)); c.execute("delete from devices where username=?", (r["username"],)); cancel_user_jobs(r["username"])
            elif a == "reset": c.execute("delete from devices where username=?", (r["username"],))
    c.commit(); c.close(); return redirect("/admin")

# ---------- iş akışı ----------
def pub(j):
    k = ("status", "phase", "progress", "frames_total", "sampled_total", "sampled_done", "speed", "eta", "live", "error", "comp_seconds", "segments", "dl")
    return {x: j.get(x) for x in k}

def own_job(u, jid):
    j = jobs.get(jid); return j if j and j["user"] == u else None

def runner(j):
    j["status"] = "queued"; j["phase"] = "Sırada bekliyor"
    with sem:
        if j["cancel"].is_set(): j["status"] = "cancelled"; return
        j["status"] = "running"; out = OUT / f"ZenithComp_{j['id']}.mp4"
        try:
            analyzer.process(j, out); j.update(status="done", progress=100, phase="Tamamlandı", output=str(out))
        except analyzer.Cancelled: j.update(status="cancelled", phase="İptal edildi")
        except Exception as e: j.update(status="error", error=str(e)[:300], phase="Hata")
        finally:
            for p in j["procs"]:
                try: p.kill()
                except Exception: pass
            try: Path(j["video"]).unlink(missing_ok=True)
            except Exception: pass

def safe_url(u):
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname: return False
    try: return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(p.hostname, None))
    except Exception: return False

def downloader(j, url):
    j["status"] = "downloading"; dest = Path(j["video"])
    try:
        if shutil.which("aria2c"):
            p = subprocess.Popen(["aria2c", "-x16", "-s16", "-k4M", "--file-allocation=none", "--allow-overwrite=true", "-d", str(dest.parent), "-o", dest.name, url],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); j["procs"].append(p)
            while p.poll() is None:
                if j["cancel"].is_set() or (dest.exists() and dest.stat().st_size > MAX_BYTES): p.kill(); break
                j["phase"] = f"Maç indiriliyor: {dest.stat().st_size / 1e9:.2f} GB" if dest.exists() else "Maç indiriliyor"; time.sleep(1)
            ok = p.returncode == 0
        else:
            import requests
            with requests.get(url, stream=True, timeout=30) as r, open(dest, "wb") as f:
                r.raise_for_status(); n = 0
                for ch in r.iter_content(4 * 1024 * 1024):
                    if j["cancel"].is_set() or n > MAX_BYTES: break
                    f.write(ch); n += len(ch); j["phase"] = f"Maç indiriliyor: {n / 1e9:.2f} GB"
            ok = not j["cancel"].is_set()
        if j["cancel"].is_set(): j["status"] = "cancelled"; return
        if not ok: raise RuntimeError("Link indirilemedi.")
        runner(j)
    except Exception as e: j.update(status="error", error=str(e)[:300])

@app.route("/api/devices")
@need_user
def devices(u): return jsonify(analyzer.list_devices())

@app.route("/api/job", methods=["POST"])
@need_user
def job_create(u):
    f = request.form; photos = request.files.getlist("photos")
    if not photos: return jsonify(error="En az 1 yüz fotoğrafı yükle."), 400
    url, fname, size = f.get("url", "").strip(), f.get("filename", ""), int(f.get("size") or 0)
    ext = (Path(urlparse(url).path).suffix.lower() if url else Path(fname).suffix.lower()) or ".mp4"
    if ext not in EXTS: ext = ".mp4" if url else ext
    if ext not in EXTS: return jsonify(error="Desteklenmeyen video türü. MP4 önerilir."), 400
    if not url and not (0 < size <= MAX_BYTES): return jsonify(error="Dosya 15 GB sınırını aşıyor."), 400
    if url and not safe_url(url): return jsonify(error="Geçersiz link."), 400
    jid = uuid.uuid4().hex[:12]; d = UP / jid; d.mkdir()
    paths = []
    for i, p in enumerate(photos[:5]):
        pp = d / f"p{i}.jpg"; p.save(pp); paths.append(str(pp))
    mode = f.get("mode", "dengeli")
    j = dict(id=jid, user=u, dir=d, video=str(d / f"video{ext}"), photos=paths, cancel=threading.Event(), procs=[], status="uploading", phase="Yükleniyor",
             progress=0, live={}, cfg=dict(device=f.get("device", "auto"), mode=mode if mode in analyzer.MODES else "dengeli", name=f.get("name", "")[:60], number=f.get("number", "")[:3]),
             size=size, got=set(), lock=threading.Lock())
    if not url:
        with open(j["video"], "wb") as fh: fh.truncate(size)
    jobs[jid] = j
    if url: threading.Thread(target=downloader, args=(j, url), daemon=True).start()
    return jsonify(id=jid, chunk=CH)

@app.route("/api/job/<jid>/chunk/<int:i>", methods=["PUT"])
@need_user
def chunk(u, jid, i):
    j = own_job(u, jid)
    if not j or j["status"] != "uploading": return jsonify(error="yok"), 404
    data = request.get_data(); off = i * CH
    if off >= j["size"] or len(data) != min(CH, j["size"] - off): return jsonify(error="bad"), 400
    with open(j["video"], "r+b") as fh: fh.seek(off); fh.write(data)
    with j["lock"]: j["got"].add(i); j["progress"] = 0
    return jsonify(ok=1)

@app.route("/api/job/<jid>/finish", methods=["POST"])
@need_user
def finish(u, jid):
    j = own_job(u, jid)
    if not j: return jsonify(error="yok"), 404
    if len(j["got"]) != -(-j["size"] // CH): return jsonify(error="Eksik parça var, tekrar dene."), 400
    threading.Thread(target=runner, args=(j,), daemon=True).start(); return jsonify(ok=1)

@app.route("/api/job/<jid>")
@need_user
def status(u, jid):
    j = own_job(u, jid)
    if not j: return jsonify(error="yok"), 404
    d = pub(j)
    if j["status"] == "done": d["dl"] = f"/api/job/{jid}/video"
    return jsonify(d)

@app.route("/api/job/<jid>/cancel", methods=["POST"])
@need_user
def cancel(u, jid):
    j = own_job(u, jid)
    if j: j["cancel"].set(); [p.kill() for p in j["procs"] if p.poll() is None]; j["status"] = "cancelled" if j["status"] != "done" else "done"
    return jsonify(ok=1)

@app.route("/api/job/<jid>/video")
@need_user
def video(u, jid):
    j = own_job(u, jid)
    if not j or j["status"] != "done": return jsonify(error="yok"), 404
    return send_file(j["output"], mimetype="video/mp4", as_attachment=request.args.get("dl") == "1", download_name="ZenithComp.mp4", conditional=True)

def cleanup():
    while True:
        for p in list(OUT.glob("*.mp4")) + list(UP.iterdir()):
            if time.time() - p.stat().st_mtime > 24 * 3600: shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        time.sleep(3600)

if __name__ == "__main__":
    from waitress import serve
    threading.Thread(target=cleanup, daemon=True).start()
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threads=48, channel_timeout=900, connection_limit=1000, max_request_body_size=70 * 1024 * 1024)
