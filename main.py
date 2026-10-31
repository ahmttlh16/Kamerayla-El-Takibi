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
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            cv2.imshow("El Tespiti", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
