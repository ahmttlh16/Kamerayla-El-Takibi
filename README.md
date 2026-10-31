# El Hareketleriyle Ses Kontrolü

Webcam görüntüsü üzerinde [MediaPipe](https://github.com/google/mediapipe) ile el tespiti yaparak; baş parmak ve işaret parmağı arasındaki mesafeyle (pinch hareketi) sistem sesini canlı olarak kontrol eden, aynı zamanda açık parmak sayısını da sayan bir OpenCV projesi.

## Özellikler

- Webcam görüntüsünde en fazla **2 el**in eş zamanlı tespiti.
- Her el için 21 eklem noktası (landmark) ve aralarındaki bağlantıların çizilmesi.
- Baş parmak (landmark 4) ile işaret parmağı (landmark 8) arasına çizilen pinch çizgisi ve orta nokta.
- Bu iki nokta arasındaki piksel mesafesinin `numpy.interp` ile ses seviyesine haritalanması.
- **Sadece ilk tespit edilen el** sistem sesini değiştirir; ekrandaki diğer el yalnızca görsel olarak takip edilir, sesi etkilemez.
- Ekranın solunda canlı güncellenen yeşil bir ses barı ve yüzdelik ses seviyesi metni.
- Tüm ellerdeki açık parmakların toplamının ekranın sol altında gösterilmesi.
- Windows'ta [pycaw](https://github.com/AndreMiras/pycaw), Linux'ta `pulsectl` veya `alsaaudio` ile işletim sistemine uygun gerçek sistem sesi kontrolü.

## Gereksinimler

- Python 3.9+
- Çalışan bir webcam

## Kurulum

```bash
# Depoyu klonla
git clone <repo-url>
cd proje_yzt

# (Onerilir) Sanal ortam olustur
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

# Bagimliliklari kur
pip install -r requirements.txt
```

> **Not:** `requirements.txt` içindeki `pycaw`/`comtypes` sadece Windows'ta, `pulsectl`/`pyalsaaudio` sadece Linux'ta kurulur (`sys_platform` koşuluyla).

### El tespiti modeli

Proje, MediaPipe'ın `HandLandmarker` modelini kullanır ve gerekli `hand_landmarker.task` dosyası depoya dahildir. Dosya bir şekilde eksikse aşağıdaki komutla tekrar indirebilirsin:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Çalıştırma

```bash
python main.py
```

- Elini kameraya göster, baş parmak ile işaret parmağını birbirine yaklaştırıp uzaklaştırarak sistem sesini ayarla.
- İkinci bir el gösterirsen o el de ekranda takip edilir ve parmak sayısına dahil edilir, ancak sesi değiştirmez.
- Çıkmak için `q` tuşuna bas.

## Proje Yapısı

```
proje_yzt/
├── main.py                # Tum uygulama mantigi tek dosyada
├── hand_landmarker.task    # MediaPipe el tespiti modeli
├── requirements.txt        # Python bagimliliklari
└── .gitignore
```

## Lisans

Bu proje eğitim/kişisel kullanım amaçlıdır; dilediğin gibi kullanabilir ve değiştirebilirsin.
