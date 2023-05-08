# Creado por Lucy
# Fecha: 05/07/2023

import cv2

def process_object_tracking():
    cap = cv2.VideoCapture(0)

    # Definir algoritmos de seguimiento
    tracker_types = ['MOSSE', 'CSRT']
    trackers = [cv2.TrackerMOSSE_create(), cv2.TrackerCSRT_create()]

    # Inicializar trackers con la región de interés del objeto en movimiento
    ret, frame = cap.read()
    rects = []
    for tracker in trackers:
        rect = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        rects.append(rect)
        cv2.destroyAllWindows()

    # Añadir la región de interés a los trackers
    for tracker, rect in zip(trackers, rects):
        tracker.init(frame, rect)

    while True:
        ret, frame = cap.read()

        for tracker in trackers:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (jpeg.tobytes())
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
