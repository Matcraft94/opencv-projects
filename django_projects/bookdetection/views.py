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

# Local application/library specific imports
from .utils import BookImageProcessor, BookDetector
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
                                    'image_path': 'book_1.png'
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

                    image_path = os.path.join(images_dir, f'book_{index}.png')
                    cv2.imwrite(image_path, book_image)
                    book.image_path = image_path
                    book.save()

                    book_ids.append(book.id)

                print(f"All books in database: {Book.objects.all()}")
                print("Books processed successfully.")
                return standard_response(
                    data={
                        "book_count": len(data),
                        "books": [{"id": book_id, "title": book.title, "image_path": book.image_path} for book_id in book_ids]
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
        books = Book.objects.all().values('id', 'title', 'author', 'image_path', 'genre__name', 'created_at')
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
