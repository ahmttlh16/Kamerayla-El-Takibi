# Mesafe hesabi icin matematik kutuphanesi
import math
# Hangi isletim sisteminde calistigimizi anlamak icin
import platform

# Kamera goruntusu ve cizim islemleri icin OpenCV
import cv2
# El tespiti modelini calistirmak icin MediaPipe
import mediapipe as mp
# Mesafe araliklarini ses/bar araligina haritalamak (map etmek) icin numpy
import numpy as np

# Modelin dosya yolunu ve calisma secenaklerini ayarlamak icin gerekli siniflar
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

# Onceden indirilen el tespiti modelinin dosya yolu
MODEL_PATH = "hand_landmarker.task"

# Eklem noktalarinin (landmark) ciziminde kullanilacak renk (BGR formatinda yesil)
LANDMARK_COLOR = (0, 255, 0)
# Eklemler arasindaki baglanti cizgilerinin rengi (BGR formatinda mavi)
CONNECTION_COLOR = (255, 0, 0)

# Bas parmak ucu (4) ve isaret parmagi ucu (8) landmark indeksleri
THUMB_TIP, INDEX_TIP = 4, 8
# Bas parmagin uc noktasi (4) ve orta eklemi (3) indeksleri
THUMB_IP = 3
# Isaret, orta, yuzuk ve serce parmaklarin (uc nokta, alt eklem) indeks ciftleri
FINGER_TIPS_PIPS = [(8, 6), (12, 10), (16, 14), (20, 18)]

# Parmaklar birbirine en yakin oldugunda (sesi kapatirken) olculen piksel mesafesi
MIN_DISTANCE = 30
# Parmaklar birbirinden en uzak oldugunda (sesi sonuna getirirken) olculen piksel mesafesi
MAX_DISTANCE = 250

# Ses barinin ekrandaki sol kenar ve genislik konumu (piksel)
BAR_X = 50
BAR_WIDTH = 35
# Ses barinin ust ve alt sinirlari (piksel) - dolum buna gore hesaplanir
BAR_TOP = 100
BAR_BOTTOM = 400


class VolumeController:
    # Isletim sistemine gore uygun sistem sesi kontrolcusunu hazirlayan sinif
    def __init__(self):
        self.system = platform.system()
        self.backend = None

        if self.system == "Windows":
            # Windows'ta pycaw kutuphanesi ile sistem sesine erisiyoruz
            from pycaw.pycaw import AudioUtilities

            speakers = AudioUtilities.GetSpeakers()
            self.backend = speakers.EndpointVolume
        elif self.system == "Linux":
            try:
                # Linux'ta PulseAudio varsa pulsectl ile sistem sesine erisiyoruz
                import pulsectl

                self.pulse = pulsectl.Pulse("el-kontrolu")
                self.sink = self.pulse.sink_list()[0]
                self.backend = "pulsectl"
            except Exception:
                # PulseAudio bulunamazsa alsaaudio ile deniyoruz
                import alsaaudio

                self.mixer = alsaaudio.Mixer()
                self.backend = "alsaaudio"
        else:
            print(f"'{self.system}' isletim sistemi icin ses kontrolu desteklenmiyor.")

    def set_volume(self, level):
        # level 0.0 (sessiz) ile 1.0 (tam ses) arasinda bir deger olmalidir
        level = max(0.0, min(1.0, level))

        if self.system == "Windows" and self.backend is not None:
            self.backend.SetMasterVolumeLevelScalar(level, None)
        elif self.backend == "pulsectl":
            self.pulse.volume_set_all_chans(self.sink, level)
        elif self.backend == "alsaaudio":
            self.mixer.setvolume(int(level * 100))


def count_fingers(hand_landmarks, handedness_label):
    # Acik parmak sayisini tutan sayac
    count = 0

    # Isaret, orta, yuzuk ve serce parmaklari tek tek kontrol et
    for tip, pip in FINGER_TIPS_PIPS:
        # Uc nokta, alt eklemden daha yukarida ise (y degeri daha kucukse) parmak aciktir
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    # Bas parmak icin yon, elin sag mi sol mu oldugu bilgisine gore degisir
    if handedness_label == "Right":
        # Sag elde bas parmak aciksa uc nokta, eklemin sagindadir (x degeri daha buyuktur)
        thumb_open = hand_landmarks[THUMB_TIP].x > hand_landmarks[THUMB_IP].x
    else:
        # Sol elde bas parmak aciksa uc nokta, eklemin solundadir (x degeri daha kucuktur)
        thumb_open = hand_landmarks[THUMB_TIP].x < hand_landmarks[THUMB_IP].x

    # Bas parmak aciksa sayaci bir arttir
    if thumb_open:
        count += 1

    # Toplam acik parmak sayisini dondur
    return count


def draw_landmarks(frame, hand_landmarks):
    # Kare (frame) yukseklik ve genislik bilgisi
    h, w = frame.shape[:2]

    # Normalize edilmis (0-1 arasi) koordinatlari piksel koordinatlarina cevir
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Eklemler arasindaki her baglanti icin bir cizgi ciz
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, CONNECTION_COLOR, 2)

    # Her eklem noktasina bir daire ciz
    for point in points:
        cv2.circle(frame, point, 5, LANDMARK_COLOR, -1)


def draw_pinch_line(frame, hand_landmarks, w, h):
    # Bas parmak ucunun piksel koordinati
    thumb_point = (
        int(hand_landmarks[THUMB_TIP].x * w),
        int(hand_landmarks[THUMB_TIP].y * h),
    )
    # Isaret parmagi ucunun piksel koordinati
    index_point = (
        int(hand_landmarks[INDEX_TIP].x * w),
        int(hand_landmarks[INDEX_TIP].y * h),
    )

    # Iki nokta arasina bir cizgi ciz
    cv2.line(frame, thumb_point, index_point, (0, 255, 255), 3)

    # Iki noktanin tam orta noktasini hesapla
    mid_point = (
        (thumb_point[0] + index_point[0]) // 2,
        (thumb_point[1] + index_point[1]) // 2,
    )
    # Orta noktaya bir daire ciz
    cv2.circle(frame, mid_point, 10, (0, 255, 255), -1)

    # Bas parmak ve isaret parmagi arasindaki mesafeyi dondur
    return math.hypot(index_point[0] - thumb_point[0], index_point[1] - thumb_point[1])


def draw_volume_bar(frame, volume_percent):
    # Ses barinin disini (bos halini) beyaz bir dikdortgen cerceve olarak ciz
    cv2.rectangle(frame, (BAR_X, BAR_TOP), (BAR_X + BAR_WIDTH, BAR_BOTTOM), (255, 255, 255), 2)

    # Yuzdeye gore barin icinin ne kadar yukaridan baslayarak dolacagini hesapla
    fill_y = int(np.interp(volume_percent, [0, 100], [BAR_BOTTOM, BAR_TOP]))
    # Barin icini yesil renkle doldur (alttan fill_y'ye kadar)
    cv2.rectangle(frame, (BAR_X, fill_y), (BAR_X + BAR_WIDTH, BAR_BOTTOM), (0, 255, 0), -1)

    # Barin altina yuzdelik ses seviyesini metin olarak yaz
    cv2.putText(
        frame,
        f"{int(volume_percent)} %",
        (BAR_X - 10, BAR_BOTTOM + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )


def main():
    # El tespiti modelinin calisma secenekleri (video modu, en fazla 2 el)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    # Sistem sesini kontrol edecek nesneyi olustur
    volume_controller = VolumeController()

    # Varsayilan webcam'i (indeks 0) ac
    cap = cv2.VideoCapture(0)

    # Kamera acilamadiysa programi sonlandir
    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    # Model kaynaklarini otomatik yonetmek icin 'with' blogu kullan
    with HandLandmarker.create_from_options(options) as landmarker:
        # Video modu icin her kareye artan bir zaman damgasi gerekir
        frame_index = 0

        while True:
            # Kameradan bir kare oku
            ret, frame = cap.read()
            if not ret:
                print("Goruntu alinamadi.")
                break

            # Goruntuyu ayna gibi gormek icin yatayda ters cevir
            frame = cv2.flip(frame, 1)
            # MediaPipe RGB formati bekledigi icin OpenCV'nin BGR formatindan donustur
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # OpenCV karesini MediaPipe'in anlayacagi goruntu nesnesine cevir
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Video modunda her karenin kendine ozgu bir zaman damgasi (ms) olmali
            timestamp_ms = int(frame_index * (1000 / 30))
            # El tespiti islemini calistir
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            # Kare boyutlarini piksel hesaplari icin al
            h, w = frame.shape[:2]

            # Tum ellerdeki acik parmaklarin toplamini tutan degisken
            toplam_parmak = 0

            # En az bir el tespit edildiyse
            if result.hand_landmarks:
                # Tespit edilen tum elleri (indeksleriyle birlikte) tek tek dolas
                for hand_index, (hand_landmarks, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    # Elin eklem noktalarini ve baglantilarini ekrana ciz
                    draw_landmarks(frame, hand_landmarks)

                    # Bas parmak ve isaret parmagi arasindaki pinch cizgisini ciz ve mesafeyi al
                    distance = draw_pinch_line(frame, hand_landmarks, w, h)

                    # Bu elin acik parmak sayisini hesapla ve toplama ekle
                    handedness_label = handedness[0].category_name
                    toplam_parmak += count_fingers(hand_landmarks, handedness_label)

                    # Ses kontrolu SADECE ilk tespit edilen el (indeks 0) icin calisir
                    if hand_index == 0:
                        # Mesafeyi ses seviyesi yuzdesine (0-100) haritala
                        volume_percent = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [0, 100])
                        # Mesafeyi ayni zamanda 0.0-1.0 araligindaki ses skalasina haritala
                        volume_scalar = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [0.0, 1.0])

                        # Hesaplanan seviyeye gore sistem sesini ayarla
                        volume_controller.set_volume(volume_scalar)

                        # Ses barini ve yuzdelik metni ekrana ciz
                        draw_volume_bar(frame, volume_percent)

                # Dongu bittikten sonra toplam acik parmak sayisini ekranin sol alt kosesine yaz
                cv2.putText(
                    frame,
                    f"Toplam Acik Parmak: {toplam_parmak}",
                    (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                )

            # Isleme sonucunu pencerede goster
            cv2.imshow("El ile Ses Kontrolu", frame)

            # 'q' tusuna basilirsa dongudan cik
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Kamerayi serbest birak
    cap.release()
    # Acik olan tum OpenCV pencerelerini kapat
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Betik dogrudan calistirildiginda ana fonksiyonu baslat
    main()
