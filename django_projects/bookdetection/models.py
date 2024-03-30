from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

    class Meta:
        app_label = 'bookdetection'

class Topic(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class Book(models.Model):
    title = models.CharField(max_length=255, unique=True)
    author = models.CharField(max_length=255, blank=True, null=True)
    main_topic = models.ForeignKey(Genre, related_name='main_topic_books', on_delete=models.SET_NULL, null=True)
    secondary_topics = models.ManyToManyField(Topic, through='BookTopic')
    created_at = models.DateTimeField(auto_now_add=True)
    digital_file_path = models.CharField(max_length=255, blank=True, null=True, help_text="Ruta al archivo digital del libro si está disponible.")
    cover_image_path = models.CharField(max_length=255, blank=True, null=True, help_text="Ruta a la imagen de la portada del libro físico.")
    book_type = models.CharField(max_length=8, choices=[('physical', 'Physical'), ('digital', 'Digital')], default='physical', help_text="Tipo de libro: físico o digital.")

    def __str__(self):
        return self.title

    def update_from_analysis(self, analysis_result):
        self.title = analysis_result.get('title', self.title)
        self.author = analysis_result.get('author', self.author)

        main_topic_name = analysis_result.get('main_topic')
        if main_topic_name:
            main_topic, _ = Genre.objects.get_or_create(name=main_topic_name)
            self.main_topic = main_topic

        # Aquí se elimina la relación de temas secundarios existentes.
        BookTopic.objects.filter(book=self).delete()

        for idx, topic_name in enumerate(analysis_result.get('secondary_topics', [])):
            topic, _ = Topic.objects.get_or_create(name=topic_name)
            confidence_level = 'Low'  # Asume un nivel de confianza por defecto o ajusta como sea necesario.
            # Si 'analysis_result' contiene niveles de confianza específicos, úsalos aquí.
            BookTopic.objects.create(book=self, topic=topic, confidence_level=confidence_level)

        self.save()


class BookTopic(models.Model):
    book = models.ForeignKey(Book, related_name='book_topics', on_delete=models.CASCADE)
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE)
    confidence_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low')])

    class Meta:
        unique_together = ('book', 'topic')

    def __str__(self):
        return f"{self.topic.name} ({self.confidence_level}) for {self.book.title}"

class Description(models.Model):
    book = models.OneToOneField(Book, related_name='description', on_delete=models.CASCADE)
    content = models.TextField()

    def __str__(self):
        return f"Description for {self.book.title}"
