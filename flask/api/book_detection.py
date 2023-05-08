# Creado por Lucy
# Fecha: 05/07/2023

import os
import re
import string
import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def save_book_image(frame, x, y, w, h, img_number):
    book_img = frame[y:y + h, x:x + w]
    cv2.imwrite(f'books_imgs/img_{img_number}.jpg', book_img)

def save_book_name(img_number, book_name):
    with open("books_names.txt", "a") as f:
        f.write(f"img_{img_number}: {book_name}\n")
        
def preprocess_text(ocr_text):
    # Eliminar saltos de línea y espacios extra
    # ocr_text = re.sub(r'\n', ' ', ocr_text)
    ocr_text = re.sub(r'\s+', ' ', ocr_text)

    # Eliminar signos de puntuación
    ocr_text = ocr_text.translate(str.maketrans("", "", string.punctuation))

    # Convertir todo el texto a minúsculas
    ocr_text = ocr_text.lower()

    return ocr_text

def extract_text(frame, x, y, w, h):
    cropped = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
    text = preprocess_text(text)
    return text.strip()


# Cargar la red YOLOv4
net = cv2.dnn.readNet(os.path.abspath("yolo/yolov4.weights"), os.path.abspath("yolo/yolov4.cfg"))
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Cargar las etiquetas y asignar colores
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_books(frame, book_count):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    conf_threshold = 0.3  # Cambiar el valor del umbral de confianza aquí
    nms_threshold = 0.5  # Cambiar el valor del umbral de supresión no máxima aquí

    # Procesar las detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == "book":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supresión no máxima
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i.item()
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Extraer y guardar el texto del libro
        extracted_text = extract_text(frame, x, y, w, h)
        text_label = f"{label}: {extracted_text}"
        cv2.putText(frame, text_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Guardar la imagen y el nombre del libro
        save_book_image(frame, x, y, w, h, book_count)
        save_book_name(book_count, extracted_text)
        book_count += 1

    return frame, book_count





def process_book_detection():
    cap = cv2.VideoCapture(0)
    book_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, book_count = detect_books(frame, book_count)
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