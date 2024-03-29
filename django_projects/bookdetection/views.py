# Created by por Lucy
# Date: 26/02/2024

# Standard library imports
import os
import re
import string

# Third-party library imports
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from fuzzywuzzy import fuzz, process
import cv2
import langid
import numpy as np
from PIL import Image
import pytesseract
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

# Django imports
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

# PyMuPDF imports
import fitz

# Local application/library specific imports
from .utils import *
from bookdetection.models import *
from shared_utils.responses import standard_response

import traceback  # Import traceback module

class BookImageProcessView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    # print(Book.objects.all())

    @swagger_auto_schema(
        operation_summary='Process and save book images',
        operation_description='Uploads an image, detects books, extracts texts, saves the images, and updates the database with the book details and image paths.',
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'image': openapi.Schema(type=openapi.TYPE_FILE, description='Image containing books')
            },
            required=['image']
        ),
        responses={
            200: openapi.Response(
                description='Books processed successfully',
                examples={
                    'application/json': {
                        'success': True,
                        'data': {
                            'book_count': 1,
                            'books': [
                                {
                                    'id': 1,
                                    'text_extracted': 'Example Book Title',
                                    'cover_image_path': 'book_1.png'
                                }
                            ]
                        },
                        'message': 'Books processed successfully.'
                    }
                }
            ),
            400: openapi.Response(
                description='No image provided',
                examples={
                    'application/json': {
                        'success': False,
                        'message': 'No image provided.'
                    }
                }
            ),
            500: openapi.Response(
                description='Error processing the image',
                examples={
                    'application/json': {
                        'success': False,
                        'message': 'An error occurred.'
                    }
                }
            )
        },
    )
    def post(self, request, *args, **kwargs):
        print("Received POST request")
        if request.method == 'POST' and request.FILES.get('image'):
            try:
                print("Processing image...")
                image_file = request.FILES['image']
                image = Image.open(image_file)
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Instancia las clases para la detección y procesamiento de la imagen
                print("Detecting books using YOLO...")
                book_detector = BookDetector()
                book_image_processor = BookImageProcessor()

                data = book_detector.detect_books_yolo(cv_image)
                print(f"Detected {len(data)} books")
                images_dir = os.path.expanduser('~/book_images')
                os.makedirs(images_dir, exist_ok=True)

                book_ids = []
                for index, book_data in enumerate(data):
                    box = book_data['box'].numpy()
                    x1, y1, x2, y2 = map(int, box[:4])
                    book_image = cv_image[y1:y2, x1:x2]

                    extracted_text = book_image_processor.extract_text(cv_image, *map(int, box))
                    truncated_text = extracted_text[:255]
                    print(f"Extracted text: '{truncated_text}'")

                    book, created = Book.objects.get_or_create(title=truncated_text, defaults={'author': 'Unknown (transcription)'})
                    print(f"Book created: {created}, Title: '{book.title}'")

                    cover_image_path = os.path.join(images_dir, f'book_{index}.png')
                    cv2.imwrite(cover_image_path, book_image)
                    book.cover_image_path = cover_image_path
                    book.book_type = 'physical'
                    book.save()

                    book_ids.append(book.id)

                print(f"All books in database: {Book.objects.all()}")
                print("Books processed successfully.")
                return standard_response(
                    data={
                        "book_count": len(data),
                        "books": [{"id": book_id, "title": book.title, "cover_image_path": book.cover_image_path} for book_id in book_ids]
                    },
                    message="Books processed successfully." if len(data) > 0 else "No books detected."
                )
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                return standard_response(
                    data=None,
                    message=f"An error occurred: {str(e)}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    success=False
                )
        else:
            print("No image provided in the request.")
            return standard_response(
                data=None,
                message="No image provided.",
                code=status.HTTP_400_BAD_REQUEST,
                success=False
            )

    def get(self, request, *args, **kwargs):
        books = Book.objects.all().values('id', 'title', 'author', 'cover_image_path',
                                          'digital_file_path', 'book_type', 'genre__name', 'created_at')
        book_list = list(books)

        # Convertir 'genre__name' a 'genre' y 'created_at' a string
        for book in book_list:
            book['genre'] = book.pop('genre__name', 'No Genre')
            book['created_at'] = book['created_at'].strftime('%Y-%m-%d %H:%M:%S') if book['created_at'] else 'Unknown'

        return standard_response(
                success= True,
                data = book_list,
                message= 'Books retrieved successfully.',
                code=status.HTTP_200_OK
            )

        
class BookPDFProcessView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if request.method != 'POST' or not request.FILES.getlist('pdf_file'):
            return standard_response(
                data={},
                message='Método no permitido o archivos PDF no proporcionados.',
                code=status.HTTP_400_BAD_REQUEST,
                success=False
            )

        pdf_files = request.FILES.getlist('pdf_file')
        response_data = []
        
        for pdf_file in pdf_files:
            file_stream = pdf_file.read()
            try:
                pdf = fitz.open(stream=file_stream, filetype="pdf")
            except Exception as e:
                response_data.append({'filename': pdf_file.name, 'success': False, 'message': str(e)})
                continue

            metadatos = pdf.metadata
            title = metadatos.get('title', 'Título desconocido')
            author = metadatos.get('author', 'Autor desconocido')

            pdf_path = os.path.expanduser(f'~/books/{pdf_file.name}')
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

            try:
                with open(pdf_path, 'wb+') as destination:
                    destination.write(file_stream)
            except Exception as e:
                response_data.append({'filename': pdf_file.name, 'success': False, 'message': str(e)})
                continue

            try:
                book, created = Book.objects.update_or_create(
                    title=title,
                    defaults={
                        'author': author,
                        'digital_file_path': pdf_path,
                        'book_type': 'digital',
                    }
                )
                # Crea una descripción genérica si no existe una ya
                Description.objects.get_or_create(
                    book=book,
                    defaults={'content': 'Descripción genérica aún no disponible.'}
                )
                response_data.append({'filename': pdf_file.name, 'success': True, 'book_id': book.id, 'message': 'Processed successfully'})
            except Exception as e:
                response_data.append({'filename': pdf_file.name, 'success': False, 'message': str(e)})

        return standard_response(data=response_data, message="Batch processing completed")


class BookDescriptionView(APIView):
    

    def post(self, request, format=None):
        try:
            book_id = request.data['book_id']
            content = request.data['content']

            # Validar que el libro existe
            book = Book.objects.get(id=book_id)

            with transaction.atomic():
                description = Description(book=book, content=content)
                description.save()

            return self.standard_response(data={"description_id": description.id}, message="Description added successfully.")

        except Book.DoesNotExist:
            return self.standard_response(message="Book not found.", code=status.HTTP_404_NOT_FOUND, success=False)
        except Exception as e:
            return self.standard_response(message=str(e), code=status.HTTP_400_BAD_REQUEST, success=False)
        

# class ExtractPDFInfoView(APIView):

#     def post(self, request, *args, **kwargs):
#         book_ids = request.data.get('book_ids')

#         if not book_ids:
#             return standard_response(
#                 data={},
#                 message='No book IDs provided',
#                 code=status.HTTP_400_BAD_REQUEST,
#                 success=False
#             )

#         response_data = []
#         ai_client = OpenAIClient()

#         for book_id in book_ids:
#             try:
#                 book = Book.objects.get(id=book_id, book_type='digital')
#             except Book.DoesNotExist:
#                 response_data.append({'error': f'Book with ID {book_id} not found or is not digital'})
#                 continue

#             pdf_path = book.digital_file_path
#             if not os.path.exists(pdf_path):
#                 response_data.append({'error': f'PDF file for book ID {book_id} not found'})
#                 continue
            
#             try:
#                 text_content = []
#                 pdf = fitz.open(pdf_path)
#                 for page in pdf:
#                     text = page.get_text().strip()
#                     if text:
#                         text_content.append(text)
#                         if len(text_content) >= 3:
#                             break
#                 pdf.close()
#             except Exception as e:
#                 response_data.append({'error': f'Failed to read PDF for book ID {book_id}: {str(e)}'})
#                 continue
            
#             # prompt = f"Extract the title, authors, main topic, and secondary topics from the following text: {' '.join(text_content)}"
#             analysis_result = ai_client.analyze_text(text_content)

#             response_data.append(analysis_result)

#         return standard_response(data=response_data, message="PDF information extraction completed")

class ExtractPDFInfoView(APIView):

    def post(self, request, *args, **kwargs):
        book_ids = request.data.get('book_ids')

        if not book_ids:
            return standard_response(
                data={},
                message='No book IDs provided',
                code=status.HTTP_400_BAD_REQUEST,
                success=False
            )

        response_data = []
        ai_client = OpenAIClient()

        for book_id in book_ids:
            with transaction.atomic():
                try:
                    book = Book.objects.select_for_update().get(id=book_id, book_type='digital')
                except Book.DoesNotExist:
                    response_data.append({'error': f'Book with ID {book_id} not found or is not digital'})
                    continue

                pdf_path = book.digital_file_path
                text_content = PDFTextExtractor.extract_text_from_pdf(pdf_path)
                if text_content is None or text_content == '':
                    response_data.append({'error': f'Failed to extract text for book ID {book_id}'})
                    continue

                analysis_result = ai_client.analyze_text(text_content)

                # Actualiza el título y autor
                book.title = analysis_result.get('title', book.title)
                book.author = analysis_result.get('author', book.author)
                book.save()

                # Actualiza el tema principal
                main_topic_name = analysis_result.get('main_topic')
                if main_topic_name:
                    main_topic, _ = Genre.objects.get_or_create(name=main_topic_name)
                    book.main_topic = main_topic
                    book.save()

                # Actualiza los temas secundarios
                BookTopic.objects.filter(book=book).delete()
                for idx, topic_name in enumerate(analysis_result.get('secondary_topics', [])):
                    topic, _ = Topic.objects.get_or_create(name=topic_name)
                    confidence_level = analysis_result['confidence_levels']['secondary_topics_confidences'][idx]
                    BookTopic.objects.create(book=book, topic=topic, confidence_level=confidence_level)

                response_data.append({'success': f'Book with ID {book_id} updated successfully', "data": analysis_result})

        return standard_response(data=response_data, message="PDF information extraction and update completed", success=True)