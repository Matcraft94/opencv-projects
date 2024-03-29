from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

import cv2
from PIL import Image

import numpy as np

import openai

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

class OpenAIClient:
    def __init__(self):
        self.load_api_key()
        self.client = openai.OpenAI()
        

    def load_api_key(self):
        try:
            with open("api_key.txt", "r") as file:
                self.api_key = file.read().strip()
                openai.api_key = self.api_key
        except Exception as e:
            print("Error loading API key:", str(e))
            self.api_key = None

    def analyze_text(self, text_content):
        try:
            if not self.api_key:
                return {'error': 'API key not loaded'}
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format="json_object",
                messages=[
                    {"role": "system", "content": """Based on the following excerpts taken from the first non-empty pages of a book in PDF format, especially those containing multiple works or an anthology, please analyze the content to identify the book's title, the names of various authors, and keywords that capture the main genre and related sub-genres or topics. Conduct a validation and cross-referencing process to ensure the accuracy of the extracted information, and estimate the confidence level for each identified piece of information (title, authors, main topic, and secondary topics). Determine the language of the text and tailor your analysis accordingly. Limit the identification of secondary topics to a maximum of ten. Here is the extracted text for your review:

Extracted text:"""},
                    {"role": "user", "content": text_content},
                    {"role": "system", "content": """After reviewing the text, please fill in the following structure with the extracted information, ensuring that 'main_topic' refers to the main genre and 'secondary_topics' include sub-genres or related topics, all within the constraints of the identified content and language(s):

{
"title": "[Insert the identified title of the book here]",
"author": [Insert a list with the names of the identified authors here]",
"main_topic": "[Insert the keyword that defines the main genre here]",
"secondary_topics": [Insert a list with up to ten keywords defining the sub-genres or related topics here]",
"confidence_levels": {
"title": "[Insert confidence level for the title here]",
"author": "[Insert confidence levels for each author here]",
"main_topic": "[Insert confidence level for the main topic here]",
"secondary_topics": "[Insert confidence levels for each secondary topic here]"
}
}"""}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return {'error': str(e)}

class PDFTextExtractor:
    """
    Una clase para extraer texto de archivos PDF sin incluir imágenes.
    """

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Esta función extrae solo el texto de un archivo PDF, preservando la estructura básica del texto.
        :param pdf_path: La ruta al archivo PDF del que se extraerá el texto.
        :return: Un string que contiene el texto extraído del PDF.
        """
        if not os.path.exists(pdf_path):
            return None  # Retorna None si el archivo no existe

        try:
            text_content = []
            pdf = fitz.open(pdf_path)
            for page in pdf:
                text_blocks = page.get_text("blocks")
                text_blocks.sort(key=lambda block: (block[1], block[0]))  # Ordena los bloques de texto
                for block in text_blocks:
                    # Cada 'block' contiene la posición del bloque en la página, el texto y otros metadatos.
                    # Aquí solo nos interesa el texto, que es el índice 4 del bloque.
                    text = block[4].strip()
                    if text:
                        text_content.append(text)
            pdf.close()
            return "\n".join(text_content)  # Une todos los fragmentos de texto con saltos de línea.
        except Exception as e:
            return f"Failed to read PDF: {str(e)}"
