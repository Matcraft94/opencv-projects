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
        # Actualizar título y autor solo si el nivel de confianza es suficiente
        if analysis_result.get('confidence_levels', {}).get('title') == 'High':
            self.title = analysis_result.get('title', self.title)

        # Considerando que podría haber varios autores con diferentes niveles de confianza
        if analysis_result.get('confidence_levels', {}).get('author', [])[0] == 'High':  # Suponiendo el primer autor como principal
            self.author = analysis_result.get('author', self.author)[0]  # Asume que el primer autor es el principal

        # Actualizar el tema principal si el nivel de confianza es alto
        if analysis_result.get('confidence_levels', {}).get('main_topic_confidence') == 'High':
            main_topic_name = analysis_result.get('main_topic')
            if main_topic_name:
                main_topic, _ = Genre.objects.get_or_create(name=main_topic_name)
                self.main_topic = main_topic

        # Eliminar las relaciones de temas secundarios existentes antes de actualizar
        BookTopic.objects.filter(book=self).delete()

        # Actualizar los temas secundarios y sus niveles de confianza
        secondary_topics = analysis_result.get('secondary_topics', [])
        secondary_topic_confidences = analysis_result.get('confidence_levels', {}).get('secondary_topics_confidences', [])

        for idx, topic_name in enumerate(secondary_topics):
            if idx < len(secondary_topic_confidences) and secondary_topic_confidences[idx] == 'High':
                topic, _ = Topic.objects.get_or_create(name=topic_name)
                confidence_level = secondary_topic_confidences[idx]
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
