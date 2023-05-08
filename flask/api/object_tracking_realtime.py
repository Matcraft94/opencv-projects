# Creado por Lucy
# Fecha: 05/07/2023

import cv2

def process_object_tracking(*args):
    cap = cv2.VideoCapture(0)

    # Crear un diccionario de trackers
    tracker_dict = {
        'MIL': cv2.legacy.TrackerMIL_create(),
        'KCF': cv2.legacy.TrackerKCF_create(),
        'TLD': cv2.legacy.TrackerTLD_create(),
        'BOOSTING': cv2.legacy.TrackerBoosting_create(),
        'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create(),
        'GOTURN': cv2.legacy.TrackerGOTURN_create(),
        'MOSSE': cv2.legacy.TrackerMOSSE_create(),
        'CSRT': cv2.legacy.TrackerCSRT_create(),
    }

    # Definir algoritmos de seguimiento en función de los argumentos proporcionados
    if len(args) == 0:
        tracker_keys = ['MOSSE']  # Utiliza MOSSE como el rastreador predeterminado si no se proporcionan argumentos
    else:
        tracker_keys = args

    trackers = [tracker_dict[key] for key in tracker_keys]

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
