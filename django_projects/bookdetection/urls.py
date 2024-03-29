# Created by por Lucy
# Date: 29/02/2024

from django.urls import path
from .views import BookImageProcessView, BookDescriptionView

app_name = 'bookdetection'

urlpatterns = [
    path('process-book-image/', BookImageProcessView.as_view(), name='process_book_image'),
    path('add-book-description/', BookImageProcessView.as_view(), name='add_book_description'),
]
