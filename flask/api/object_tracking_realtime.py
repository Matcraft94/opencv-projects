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
    for i, tracker in enumerate(trackers):
        tracker.init(frame, rects[i])

    while True:
        ret, frame = cap.read()

        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, tracker_types[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
