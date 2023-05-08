# Creado por Lucy
# Fecha: 05/07/2023

import cv2
import imutils


def process_object_tracking(*tracker_types):
    # Crear un diccionario de trackers
    tracker_dict = {
        'MIL': cv2.legacy.TrackerMIL_create(),
        'KCF': cv2.legacy.TrackerKCF_create(),
        'TLD': cv2.legacy.TrackerTLD_create(),
        'BOOSTING': cv2.legacy.TrackerBoosting_create(),
        'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create(),
        'MOSSE': cv2.legacy.TrackerMOSSE_create(),
        'CSRT': cv2.legacy.TrackerCSRT_create(),
    }

    # Diccionario de colores para cada método de seguimiento
    tracker_colors = {
        'MIL': (0, 255, 0),
        'KCF': (255, 0, 0),
        'TLD': (0, 0, 255),
        'BOOSTING': (255, 255, 0),
        'MEDIANFLOW': (0, 255, 255),
        'MOSSE': (255, 0, 255),
        'CSRT': (255, 255, 255),
    }

    # Inicializar los trackers
    trackers = cv2.legacy.MultiTracker_create()
    cap = cv2.VideoCapture(0)

    # Seleccionar múltiples regiones en la primera captura
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        return iter(()) # Devolver un objeto iterable vacío si no se pudo leer el frame

    for tracker_type in tracker_types:
        bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker = tracker_dict[tracker_type]
        trackers.add(tracker, frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (success, boxes) = trackers.update(frame)
        for i, box in enumerate(boxes):
            if success:
                (x, y, w, h) = [int(v) for v in box]
                tracker_type = tracker_types[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), tracker_colors[tracker_type], 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
