from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

import os
import json

import cv2
from PIL import Image

import numpy as np

import openai
import tiktoken

import pytesseract

import fitz

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
    def __init__(self, model="gpt-3.5-turbo"):
        self.load_api_key()
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = model
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                print(f"Model {model} not found in tiktoken. Falling back to a default encoding.")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            raise Exception("API key not loaded. Make sure api_key.txt is present and accessible.")
    
    def num_tokens_from_string(self, string: str) -> int:
        """
        Retorna el número de tokens en una cadena de texto utilizando la codificación para el modelo específico.
        """
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def num_tokens_from_messages(self, messages):
        """
        Retorna el número de tokens usados por una lista de mensajes, ajustando para el formato de mensajes de Chat de OpenAI.
        """
        tokens_per_message = 3  # Asumir un valor predeterminado, ajustable según el modelo.
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "role":
                    # El nombre del rol podría requerir ajustes de tokens adicionales.
                    num_tokens += 1
        # Ajuste para el priming de cada respuesta con el asistente.
        num_tokens += 3
        return num_tokens

    def trim_to_max_tokens(self, text_content: str, max_tokens: int = 1500) -> str:
        """
        Recorta el contenido del texto a un número máximo de tokens.
        """
        tokens = self.encoding.encode(text_content)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            trimmed_text = self.encoding.decode(tokens)
            return trimmed_text
        else:
            return text_content

    def load_api_key(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        api_key_file_path = os.path.join(current_dir, "api_key.txt")

        try:
            with open(api_key_file_path, "r") as file:
                self.api_key = file.read().strip()
                print(f"API key loaded successfully: {self.api_key[:4]}...{self.api_key[-4:]}")
        except IOError as e:
            print(f"Error loading API key: {e}")
            self.api_key = None

    def analyze_text(self, text_content):
        trimmed_text_content = self.trim_to_max_tokens(text_content, 1500)
#         messages = [
#                     {"role": "system", "content": """Based on the following excerpts taken from the first non-empty pages of a book in PDF format, especially those containing multiple works or an anthology, please analyze the content to identify the book's title, the names of various authors, and keywords that capture the main genre and related sub-genres or topics. Conduct a validation and cross-referencing process to ensure the accuracy of the extracted information, and estimate the confidence level for each identified piece of information (title, authors, main topic, and secondary topics). Determine the language of the text and tailor your analysis accordingly. Limit the identification of secondary topics to a maximum of ten. Here is the extracted text for your review:

# Extracted text:"""},
#                     {"role": "user", "content": trimmed_text_content},
#                     {"role": "system", "content": """After reviewing the text, please fill in the following structure with the extracted information, ensuring that 'main_topic' refers to the main genre and 'secondary_topics' include sub-genres or related topics, all within the constraints of the identified content and language(s):

# {
# "title": "[Insert the identified title of the book here]",
# "author": [Insert a list with the names of the identified authors here]",
# "main_topic": "[Insert the keyword that defines the main genre here]",
# "secondary_topics": [Insert a list with up to ten keywords defining the sub-genres or related topics here]",
# "confidence_levels": {
# "title": "[Insert confidence level for the title here]",
# "author": "[Insert confidence levels for each author here]",
# "main_topic_confidence": "[Insert confidence level for the main topic here]",
# "secondary_topics_confidences": "[Insert confidence levels for each secondary topic here]"
# }
# }"""}
#                 ]
        messages = [
            {"role": "system", "content": """You will analyze the excerpts from the first non-empty pages of a book, likely containing multiple works or an anthology. Your task is to identify the book's title, the names of various authors, and keywords that capture the main genre and related sub-genres or topics based on the content provided. Please execute a validation and cross-referencing process to ensure the accuracy of the identified information, estimating the confidence level for each identified piece, including the title, authors, main topic, and up to ten secondary topics. Determine the language of the text and adjust your analysis accordingly. You need to populate the given structure with 'main_topic_confidence' and 'secondary_topics_confidences' as lists, reflecting the certainty of each identified genre and topic. Here is the structure to be filled based on your analysis:
             {
            "title": "[Identified title]",
            "author": "[List of authors]",
            "main_topic": "[Main genre keyword]",
            "secondary_topics": "[List of up to ten sub-genres or related topics]",
            "confidence_levels": {
                "title": "[Confidence level for the title]",
                "author": "[Confidence levels for each author]",
                "main_topic_confidence": "[Confidence level for the main topic]",
                "secondary_topics_confidences": "[Confidence levels for each secondary topic]"
            }
            }
            """},
            {"role":"user", "content":trimmed_text_content},
        ]
        try:
            if not self.api_key:
                return {'error': 'API key not loaded'}
            
            # Intento de llamar a la API y usar la respuesta.
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                # response_format="json",
                messages=messages
            )

            # print("\n\n\n-----------------------------")
            num_tokens = self.num_tokens_from_messages(messages)
            # print(f"Total tokens to be used: {num_tokens}")
            # print(response.choices[0].message.content)
            return json.loads(response.choices[0].message.content)

        except openai.BadRequestError as e:
            # This catches HTTP errors related to bad requests specifically.
            print("Bad request to OpenAI:", e)
            return {'error': 'Bad request error: ' + str(e)}
        except Exception as e:
            # This catches any other generic errors.
            print("An unexpected error occurred:", e)
            return {'error': str(e)}

class PDFTextExtractor:
    """
    Una clase para extraer texto de archivos PDF sin incluir imágenes, 
    procesando un máximo de 10 páginas o el total de páginas si son menos de 10.
    """

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Esta función extrae el texto de las primeras 10 páginas de un archivo PDF o menos si el PDF tiene menos de 10 páginas,
        preservando la estructura básica del texto.
        :param pdf_path: La ruta al archivo PDF del que se extraerá el texto.
        :return: Un string que contiene el texto extraído del PDF.
        """
        if not os.path.exists(pdf_path):
            return None  # Retorna None si el archivo no existe.

        try:
            text_content = []
            pdf = fitz.open(pdf_path)
            # Procesa hasta 10 páginas o el total de páginas si son menos de 10.
            num_pages = min(10, len(pdf))
            for page_number in range(num_pages):
                page = pdf.load_page(page_number)
                text_page = page.get_text("text")
                if text_page:
                    text_content.append(text_page)
            pdf.close()
            return "\n".join(text_content)  # Une todos los fragmentos de texto con saltos de línea.
        except Exception as e:
            return f"Failed to read PDF: {str(e)}"
