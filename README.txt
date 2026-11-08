ZenithCompMaker kurulum
1) Python 3.10+ ve NVENC destekli ffmpeg kur (ffmpeg -encoders | grep nvenc). Hızlı link indirme için aria2c önerilir.
2) pip install -r requirements.txt   (torch'u CUDA sürümüyle kur: pytorch.org)
3) python app.py   -> http://SUNUCU:5000
Sunucuda NVIDIA GPU (T4/L4/RTX) olmalı. Admin şifresini değiştirmek için: ZENITH_ADMIN_PASSWORD=yenisifre python app.py
Veriler data/ klasöründe; çıktılar 24 saat sonra silinir.

Sunucu (kalıcı link): sunucuya zip'i yükle, unzip et, `bash deploy.sh` çalıştır. Servis olarak başlar, yeniden başlatmada otomatik açılır.
Güncelleme: dosyaları değiştir, `sudo systemctl restart zenith`. Log: `journalctl -u zenith -f`

GPU'suz sunucu: deploy.sh GPU yoksa otomatik CPU sürümünü kurar. CPU'da "Hızlı" modu kullan; çekirdek sayısı arttıkça hızlanır (8+ çekirdek önerilir).

Kontrol: venv/bin/python selftest.py  (GPU, NVENC, CUDA, paketler)
Modlar: Hızlı/Dengeli=YOLOv8m, Detaylı=YOLOv8l, Maksimum=YOLOv8x (GPU'da). Önerilen GPU: RTX 4090 / L4 / A10 / A100.
Docker pod (RunPod/Vast): systemd yoksa deploy.sh siteyi arka planda başlatır; pod'un 5000 portunu dışarı aç (HTTP port).
