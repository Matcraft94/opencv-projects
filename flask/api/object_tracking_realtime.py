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
        # 'GOTURN': cv2.legacy.TrackerGOTURN_create(),
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
    tracker_list = []
    cap = cv2.VideoCapture(0)

    # Seleccionar múltiples regiones en la primera captura
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        return iter(())  # Devolver un objeto iterable vacío si no se pudo leer el frame

    for tracker_type in tracker_types:
        bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker = tracker_dict[tracker_type].create()  # Creamos el tracker usando create()
        tracker.init(frame, bbox)
        tracker_list.append(tracker)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tracker_type, tracker in zip(tracker_types, tracker_list):
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), tracker_colors[tracker_type], 2)

        frame = imutils.resize(frame, width=800)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()