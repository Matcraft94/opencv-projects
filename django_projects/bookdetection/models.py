from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name
    
    class Meta:
        app_label = 'bookdetection'

class Book(models.Model):
    title = models.CharField(max_length=255, unique=True)
    author = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    genre = models.ForeignKey(Genre, on_delete=models.SET_NULL, null=True)

    digital_file_path = models.CharField(max_length=255, blank=True, null=True, help_text="Ruta al archivo digital del libro si está disponible.")

    cover_image_path = models.CharField(max_length=255, blank=True, null=True, help_text="Ruta a la imagen de la portada del libro físico.")

    TYPE_CHOICES = (
        ('physical', 'Physical'),
        ('digital', 'Digital'),
    )
    book_type = models.CharField(max_length=8, choices=TYPE_CHOICES, default='physical', help_text="Tipo de libro: físico o digital.")

    def __str__(self):
        return self.title

class Description(models.Model):
    book = models.OneToOneField(Book, on_delete=models.CASCADE)
    content = models.TextField()

    def __str__(self):
        return f"Description for {self.book.title}"
