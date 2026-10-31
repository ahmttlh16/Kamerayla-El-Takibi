import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

MODEL_PATH = "hand_landmarker.task"

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)

THUMB_TIP, THUMB_IP = 4, 3
FINGER_TIPS_PIPS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def count_fingers(hand_landmarks, handedness_label):
    count = 0

    for tip, pip in FINGER_TIPS_PIPS:
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    if handedness_label == "Right":
        thumb_open = hand_landmarks[THUMB_TIP].x > hand_landmarks[THUMB_IP].x
    else:
        thumb_open = hand_landmarks[THUMB_TIP].x < hand_landmarks[THUMB_IP].x

    if thumb_open:
        count += 1

    return count


def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]

    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, CONNECTION_COLOR, 2)

    for point in points:
        cv2.circle(frame, point, 5, LANDMARK_COLOR, -1)


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Goruntu alinamadi.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(frame_index * (1000 / 30))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    draw_landmarks(frame, hand_landmarks)

                    finger_count = count_fingers(
                        hand_landmarks, handedness[0].category_name
                    )
                    tip = hand_landmarks[0]
                    text_pos = (int(tip.x * frame.shape[1]) - 20, int(tip.y * frame.shape[0]) + 40)
                    cv2.putText(
                        frame,
                        str(finger_count),
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                    )

            cv2.imshow("El Tespiti", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
