from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

import cv2
from PIL import Image

import numpy as np

import pytesseract

import re
import string
from fuzzywuzzy import fuzz, process
import langid

class ImageUtils:
    def normalize_color(color):
        return [element / 255 for element in color]

    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def adaptive_threshold(image):
        gray_image = ImageUtils.to_grayscale(image)
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def preprocess_for_ocr(image, blur_radius=2):
        gray_image = ImageUtils.to_grayscale(image)
        blurred_image = cv2.GaussianBlur(gray_image, (blur_radius, blur_radius), 0)
        return cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

class BookImageProcessor:
    # def __init__(self, net, classes, colors):
    #     self.net = net
    #     self.classes = classes
    #     self.colors = colors

    @staticmethod
    def preprocess_text(ocr_text):
        ocr_text = re.sub(r'\s+', ' ', ocr_text)
        ocr_text = ocr_text.translate(str.maketrans("", "", string.punctuation))
        ocr_text = ocr_text.lower()
        return ocr_text

    def extract_text(self, frame, x, y, w, h):
        # Recortar la imagen
        cropped = frame[y:y + h, x:x + w]
        
        # Mejorar el contraste y convertir a escala de grises
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Aplicar filtro Gaussiano para reducir el ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Usar umbralización adaptativa
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Extraer texto con Tesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, lang='eng', config=custom_config)
        
        # Preprocesar el texto extraído
        return self.preprocess_text(text)

    def save_book_image(self, frame, x, y, w, h, img_number):
        book_img = frame[y:y + h, x:x + w]
        path = f'books_imgs/img_{img_number}.jpg'
        cv2.imwrite(path, book_img)
        return path

class BookDetector:
    # def __init__(self, processor, conf_threshold):
    #     self.processor = processor
    #     self.conf_threshold = conf_threshold

    # def get_output_layers(self, net):
    #     layer_names = net.getLayerNames()
    #     # Cambio para la compatibilidad con OpenCV 4.x: elimina el [0] - 1
    #     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #     return output_layers

    def extract_box(self, detection, width, height):
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        x = center_x - w // 2
        y = center_y - h // 2
        return [x, y, w, h]

    def detect_books_yolo(self, frame):
        track_history = defaultdict(lambda: [])
        model = YOLO("yolov8n.pt")
        names = model.model.names

        results = model.track(frame, persist=True, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        # Mapear los índices de clase a nombres de clase
        clss_names = [names[cls] for cls in clss]

        data = [{'box': box, 'conf': conf,'cls_name': cls_name, 'track_id': track_id}
            for box, conf, cls_name, track_id in zip(boxes, confs, clss_names, track_ids) if cls_name == 'book']
        return data


        


    # def detect_books(self, frame, book_count, text_extracted_books, classes):
    #     height, width, channels = frame.shape
    #     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #     self.processor.net.setInput(blob)
    #     output_layers = self.get_output_layers(self.processor.net)
    #     layer_outputs = self.processor.net.forward(output_layers)

    #     book_images = []
    #     boxes = []

    #     for output in layer_outputs:
    #         for detection in output:
    #             scores = detection#[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
    #             if 'book' == classes[class_id]:
    #                 print(f"Detected class: {classes[class_id]}, Confidence: {confidence:.2f}")
    #             if confidence > self.conf_threshold and classes[class_id] == "book":
    #                 box = self.extract_box(detection, width, height)
    #                 boxes.append(box)
    #                 x, y, w, h = box
    #                 extracted_text = self.processor.extract_text(frame, x, y, w, h)
    #                 if extracted_text and langid.classify(extracted_text)[0] == 'es':
    #                     book_image = frame[y:y+h, x:x+w]
    #                     book_images.append(book_image)
    #                     book_count += 1
    #                     text_extracted_books.append({'extracted_text': extracted_text, 'image_path': None})

    #     # images_array = np.array(book_images)

    #     return frame, book_count, text_extracted_books, book_images, boxes
