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
from django.db import transaction, IntegrityError

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
                    # print(f"Extracted text: '{truncated_text}'")

                    # Asigna un género principal por simplicidad
                    genre, _ = Genre.objects.get_or_create(name="Unknown Genre")
                    book, created = Book.objects.get_or_create(
                        title=truncated_text, 
                        defaults={'author': 'Unknown (transcription)', 'main_topic': genre}
                    )
                    print(f"Book created: {created}, Title: '{book.title}'")

                    cover_image_path = os.path.join(images_dir, f'book_{index}.png')
                    cv2.imwrite(cover_image_path, book_image)
                    book.cover_image_path = cover_image_path
                    book.save()

                    # Crea una descripción de ejemplo para el libro
                    Description.objects.get_or_create(
                        book=book,
                        defaults={'content': 'No description available.'}
                    )

                    book_ids.append(book.id)

                # print(f"All books in database: {Book.objects.all()}")
                print("Books processed successfully.")
                return standard_response(
                    data={
                        "book_count": len(data),
                        "books": [
                            {"id": book_id, "title": book.title, "cover_image_path": book.cover_image_path} 
                            for book_id in book_ids
                        ]
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
        books = Book.objects.all().values(
            'id', 'title', 'author', 'cover_image_path',
            'digital_file_path', 'book_type', 'main_topic__name', 'created_at'
        )

        book_list = list(books)

        # Asegúrate de cambiar 'main_topic__name' a 'genre' correctamente y formatear 'created_at'.
        for book in book_list:
            # Cambia 'main_topic__name' a 'genre'
            book['genre'] = book.pop('main_topic__name', 'No Genre')
            # Formatea 'created_at' como una cadena en un formato legible.
            book['created_at'] = book['created_at'].strftime('%Y-%m-%d %H:%M:%S') if book['created_at'] else 'Unknown'

        return standard_response(
            success=True,
            data=book_list,
            message='Books retrieved successfully.'
        )

        
class BookPDFProcessView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    @swagger_auto_schema(
    operation_summary='Process and save PDF books',
    operation_description='Processes uploaded PDF files, extracts metadata, saves the files, and updates the database with book details.',
    request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'pdf_file': openapi.Schema(type=openapi.TYPE_FILE, description='PDF file to process')
            },
            required=['pdf_file']
        ),
        responses={
            200: openapi.Response(
                description='PDFs processed successfully',
                examples={
                    'application/json': {
                        'success': True,
                        'data': [
                            {
                                'filename': 'example.pdf',
                                'success': True,
                                'book_id': 1,
                                'message': 'Processed successfully'
                            }
                        ],
                        'message': 'Batch processing completed'
                    }
                }
            ),
            400: openapi.Response(
                description='Invalid request or no PDF provided',
                examples={
                    'application/json': {
                        'success': False,
                        'message': 'Method not allowed or PDF files not provided.'
                    }
                }
            ),
            500: openapi.Response(
                description='Error processing the PDFs',
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
        print("Inicio del método POST.")

        if request.method != 'POST' or not request.FILES.getlist('pdf_file'):
            print("No es un método POST o no hay archivos PDF.")
            return standard_response(
                data={},
                message='Método no permitido o archivos PDF no proporcionados.',
                code=status.HTTP_400_BAD_REQUEST,
                success=False
            )

        pdf_files = request.FILES.getlist('pdf_file')
        print(f"Archivos recibidos: {pdf_files}")

        response_data = []

        for pdf_file in pdf_files:
            print(f"Procesando: {pdf_file.name}")
            try:
                pdf_file.seek(0)
                file_stream = pdf_file.read()

                with fitz.open(stream=file_stream, filetype="pdf") as pdf:
                    metadata = pdf.metadata
                    title = metadata.get('title', pdf_file.name[:-4]).strip() or "Untitled Document"
                    author = metadata.get('author', 'Autor desconocido').strip() or "Unknown Author"

                pdf_path = os.path.expanduser(f'~/books/{pdf_file.name}')
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

                print(f"Guardando PDF en: {pdf_path}")
                pdf_file.seek(0)
                with open(pdf_path, 'wb+') as destination:
                    for chunk in pdf_file.chunks():
                        destination.write(chunk)

                with transaction.atomic():
                    book, created = Book.objects.get_or_create(
                        title=title,
                        defaults={'author': author, 'book_type': 'digital', 'digital_file_path': pdf_path}
                    )

                    if not created:
                        # El libro ya existe, así que se actualiza la información.
                        book.digital_file_path = pdf_path
                        book.author = author
                        book.book_type = 'digital'
                        book.save()
                        
                    Description.objects.get_or_create(book=book, defaults={'content': 'Unknown description.'})

                    print(f"Libro procesado: {book.title}, ID: {book.id}")
                    response_data.append({
                        'filename': pdf_file.name, 
                        'success': True, 
                        'book_id': book.id, 
                        'message': 'Processed successfully'
                    })

            except Exception as e:
                print(f"Error al procesar {pdf_file.name}: {e}")
                traceback.print_exc()
                response_data.append({
                    'filename': pdf_file.name, 
                    'success': False, 
                    'message': str(e)
                })

        print("Procesamiento completado.")
        return standard_response(data=response_data, message="Batch processing completed")


class BookDescriptionView(APIView):
    
    @swagger_auto_schema(
    operation_summary='Add a description to a book',
    operation_description='Adds a descriptive text to the specified book in the database.',
    request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'book_id': openapi.Schema(type=openapi.TYPE_INTEGER, description='ID of the book to describe'),
                'content': openapi.Schema(type=openapi.TYPE_STRING, description='Description content')
            },
            required=['book_id', 'content']
        ),
        responses={
            200: openapi.Response(
                description='Description added successfully',
                examples={
                    'application/json': {
                        'success': True,
                        'data': {
                            'description_id': 1
                        },
                        'message': 'Description added successfully.'
                    }
                }
            ),
            400: openapi.Response(
                description='Invalid request or error adding description',
                examples={
                    'application/json': {
                        'success': False,
                        'message': 'An error occurred or book not found.'
                    }
                }
            )
        },
    )
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
    @swagger_auto_schema(
    operation_summary='Extract information from PDF books',
    operation_description='Extracts and updates information from specified PDF book IDs, including titles, authors, and topics.',
    request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'book_ids': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_INTEGER), description='Array of book IDs to process')
            },
            required=['book_ids']
        ),
        responses={
            200: openapi.Response(
                description='PDF information extracted and updated successfully',
                examples={
                    'application/json': {
                        'success': True,
                        'data': [
                            {
                                'success': 'Book with ID 1 updated successfully',
                                'data': {
                                    'title': 'Example Book Title',
                                    'author': ['Author One', 'Author Two'],
                                    'main_topic': 'Example Topic',
                                    'secondary_topics': ['Subtopic 1', 'Subtopic 2'],
                                    'confidence_levels': {
                                        'title': 'High',
                                        'author': 'Medium',
                                        'main_topic_confidence': 'High',
                                        'secondary_topics_confidences': ['High', 'Medium']
                                    }
                                }
                            }
                        ],
                        'message': 'PDF information extraction and update completed'
                    }
                }
            ),
            400: openapi.Response(
                description='Invalid request or book IDs not provided',
                examples={
                    'application/json': {
                        'success': False,
                        'message': 'No book IDs provided.'
                    }
                }
            ),
            500: openapi.Response(
                description='Error extracting information from PDFs',
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
        book_ids = request.data.get('book_ids')

        if not book_ids:
            print("No se proporcionaron IDs de libros.")
            return standard_response(
                data={},
                message='No book IDs provided',
                code=status.HTTP_400_BAD_REQUEST,
                success=False
            )

        response_data = []
        ai_client = OpenAIClient()

        for book_id in book_ids:
            print(f"Procesando el libro con ID: {book_id}")
            with transaction.atomic():
                try:
                    book = Book.objects.select_for_update().get(id=book_id, book_type='digital')
                except Book.DoesNotExist:
                    print(f"Libro con ID {book_id} no encontrado o no es digital.")
                    response_data.append({'error': f'Book with ID {book_id} not found or is not digital'})
                    continue

                try:
                    text_content = PDFTextExtractor.extract_text_from_pdf(book.digital_file_path)
                    if not text_content:
                        raise ValueError(f'No text content extracted for book ID {book_id}')

                    analysis_result = ai_client.analyze_text(text_content)
                    book.update_from_analysis(analysis_result)

                    # Renombrar el archivo PDF
                    if book.author and book.title:
                        new_filename = f"{book.author.replace(' ', '_')}-{book.title.replace(' ', '_')}.pdf"
                        new_filepath = os.path.join(os.path.dirname(book.digital_file_path), new_filename)
                        os.rename(book.digital_file_path, new_filepath)
                        book.digital_file_path = new_filepath
                        book.save()

                    response_data.append({'success': f'Book with ID {book_id} updated and renamed successfully', "data": analysis_result})

                except Exception as e:
                    print(f"Error al procesar el libro con ID {book_id}: {e}")
                    traceback.print_exc()
                    response_data.append({'error': f'Error processing book ID {book_id}'})

        print("Extracción y actualización de información de PDF completadas, incluido el renombrado de archivos.")
        return standard_response(data=response_data, message="PDF information extraction, update, and renaming completed", success=True)