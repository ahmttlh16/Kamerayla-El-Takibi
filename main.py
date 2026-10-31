# Kamera goruntusu ve cizim islemleri icin OpenCV
import cv2

# El tespiti modelini calistirmak icin MediaPipe
import mediapipe as mp

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

# Bas parmagin uc noktasi (4) ve orta eklemi (3) indeksleri
THUMB_TIP, THUMB_IP = 4, 3
# Isaret, orta, yuzuk ve serce parmaklarin (uc nokta, alt eklem) indeks ciftleri
FINGER_TIPS_PIPS = [(8, 6), (12, 10), (16, 14), (20, 18)]


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


def main():
    # El tespiti modelinin calisma secenekleri (video modu, en fazla 2 el)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

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

            # Bu karedeki toplam acik parmak sayisi (tum eller dahil)
            total_fingers = 0

            # En az bir el tespit edildiyse
            if result.hand_landmarks:
                # Tespit edilen her el icin eklemleri ve parmak sayisini isle
                for hand_landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    # Elin eklem noktalarini ve baglantilarini ekrana ciz
                    draw_landmarks(frame, hand_landmarks)

                    # Bu elin acik parmak sayisini hesapla
                    finger_count = count_fingers(
                        hand_landmarks, handedness[0].category_name
                    )
                    # Tum ellerin parmak sayilarini topla
                    total_fingers += finger_count

            # Toplam parmak sayisini yazdirmadan once okunakli olmasi icin arka plana koyu bir dikdortgen ciz
            cv2.rectangle(frame, (10, 10), (220, 110), (0, 0, 0), -1)
            # Toplam acik parmak sayisini ekranin sol ust kosesine buyuk ve beyaz fontla yaz
            cv2.putText(
                frame,
                f"Parmak: {total_fingers}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 255, 255),
                4,
            )

            # Isleme sonucunu pencerede goster
            cv2.imshow("El Tespiti", frame)

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
