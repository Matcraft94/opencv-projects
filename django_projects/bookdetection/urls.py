# Created by por Lucy
# Date: 29/02/2024

from django.urls import path
from .views import BookImageProcessView, BookDescriptionView, BookPDFProcessView, ExtractPDFInfoView

app_name = 'bookdetection'

urlpatterns = [
    path('process-book-image/', BookImageProcessView.as_view(), name='process_book_image'),
    path('add-book-description/', BookDescriptionView.as_view(), name='add_book_description'),
    path('process-book-pdf/', BookPDFProcessView.as_view(), name='process_book_pdf'),
    path('extract-pdf-info/', ExtractPDFInfoView.as_view(), name='extract_pdf_info'),
]
