"""Sunucu kontrolü: python selftest.py"""
import shutil, subprocess
def ok(n, v, hint=""): print(("[OK]  " if v else "[!!]  ") + n + ("" if v else f"  -> {hint}"))
ok("ffmpeg/ffprobe", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "ffmpeg kur")
enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout if shutil.which("ffmpeg") else ""
ok("NVENC (GPU kodlama)", "h264_nvenc" in enc, "deploy.sh NVENC'li ffmpeg indirir; CPU kodlamaya düşer")
hw = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout if shutil.which("ffmpeg") else ""
ok("NVDEC/cuda çözme", "cuda" in hw, "CPU çözmeye düşer")
ok("aria2c (hızlı link indirme)", bool(shutil.which("aria2c")), "apt install aria2")
try:
    import torch
    ok("torch CUDA", torch.cuda.is_available(), "GPU yok veya sürücü/torch uyumsuz (CPU modunda çalışır)")
    for i in range(torch.cuda.device_count()): print("      GPU", i, torch.cuda.get_device_name(i))
except Exception as e: ok("torch", False, str(e))
try:
    import onnxruntime as ort; p = ort.get_available_providers(); ok("onnxruntime CUDA (yüz tanıma)", "CUDAExecutionProvider" in p, f"sağlayıcılar: {p}")
except Exception as e: ok("onnxruntime", False, str(e))
for m in ("ultralytics", "insightface", "easyocr", "flask", "waitress"):
    try: __import__(m); ok(m, True)
    except Exception as e: ok(m, False, str(e))
