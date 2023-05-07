# Creado por Lucy
# Fecha: 05/07/2023

import cv2
import mediapipe as mp

def process_pose_estimation():
    cap = cv2.VideoCapture(0)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield frame
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
