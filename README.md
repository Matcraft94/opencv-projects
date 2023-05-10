# Computer Vision Projects con OpenCV y MediaPipe

Este repositorio contiene cuatro proyectos de visión por computadora utilizando OpenCV y MediaPipe en Python. Los proyectos cubren la creación de imágenes panorámicas, el seguimiento de objetos en movimiento en videos, la estimación de postura humana en videos y la detección de libros en estanterías.

## Proyecto 1: Creación de imagen panorámica

El objetivo de este proyecto es combinar varias imágenes de una habitación o panorama para crear una imagen panorámica completa. El proceso consiste en encontrar características únicas en cada imagen, emparejar las características entre las imágenes, transformar y alinear las imágenes utilizando homografía y, finalmente, combinar las imágenes.

Pasos del proyecto:
1. Tomar fotografías de un panorama con la cámara montada en un trípode.
2. Encontrar características únicas en cada imagen utilizando un algoritmo de detección de características, como SIFT, SURF o ORB.
3. Realizar un emparejamiento de características entre las imágenes utilizando técnicas como BFMatcher o FLANN.
4. Utilizar la homografía para transformar y alinear las imágenes en una única imagen panorámica.
5. Combinar las imágenes utilizando una técnica de fusión, como la fusión de imágenes basada en pirámides.
6. Presentar los resultados en un Jupyter Notebook en formato PDF.

Mejoras posibles:
- Probar diferentes algoritmos de detección de características y emparejamiento para obtener resultados más precisos.
- Implementar una función de fusión de imágenes más avanzada para reducir las inconsistencias en las áreas de solapamiento.

## Proyecto 2: Seguimiento de objetos en movimiento en un video

El objetivo de este proyecto es realizar el seguimiento de un objeto en movimiento en un video utilizando al menos dos algoritmos de seguimiento de objetos. Selecciona un video corto que capture un objeto en movimiento y define una región de interés que incluya el objeto. Implementa y compara al menos dos algoritmos de seguimiento de objetos, como MOSSE, KCF o CSRT. Muestra la trayectoria del objeto en el video y en un gráfico XY.

Pasos del proyecto:
1. Seleccionar un video corto que capture un objeto en movimiento.
2. Cargar el video utilizando OpenCV.
3. Definir una región de interés que incluya el objeto.
4. Implementar al menos dos algoritmos de seguimiento de objetos (por ejemplo, MOSSE, KCF o CSRT).
5. Mostrar la trayectoria del objeto en el video y en un gráfico XY.
6. Visualizar ambos videos resultantes con el objeto de interés resaltado.

Mejoras posibles:
- Probar diferentes algoritmos de seguimiento de objetos para ver cuál funciona mejor en diferentes condiciones.
- Implementar un sistema de votación entre múltiples algoritmos de seguimiento para obtener un seguimiento más robusto.

## Proyecto 3: Estimación de postura humana en un video

El objetivo de este proyecto es estimar la postura humana en un video utilizando el algoritmo de MediaPipe. MediaPipe es una biblioteca de aprendizaje automático que proporciona soluciones para la estimación de postura en tiempo real. Los modelos de MediaPipe son capaces de detectar y localizar puntos clave del cuerpo humano, comola cabeza, el torso, los brazos y las piernas. Al dibujar estos puntos clave y sus conexiones en cada frame del video, es posible visualizar y analizar la postura humana en el video resultante.

Pasos del proyecto:
1. Seleccionar un video que capture movimiento humano.
2. Cargar el video utilizando OpenCV.
3. Utilizar el algoritmo de MediaPipe para encontrar las partes importantes del cuerpo humano, como la cabeza, el torso, los brazos y las piernas.
4. Dibujar los landmarks y las conexiones entre ellos en cada frame del video utilizando OpenCV.
5. Visualizar el video resultante con los landmarks y conexiones dibujados.

Mejoras posibles:
- Implementar la estimación de postura en tiempo real utilizando una cámara web en lugar de un video pregrabado.
- Utilizar los datos de landmarks para calcular ángulos y distancias entre partes del cuerpo y realizar análisis de movimiento más avanzados.
- Combinar la estimación de postura con otros modelos de MediaPipe, como la estimación de manos o rostros, para una visión más completa del movimiento humano.

## Proyecto 4: Detección de libros en estanterías

El objetivo de este proyecto es detectar libros en una estantería y reconocer sus títulos mediante OCR (Reconocimiento óptico de caracteres). La aplicación se basa en OpenCV para la detección de bordes y contornos de libros y utiliza Tesseract para extraer texto de las imágenes.

Pasos del proyecto:
1. Tomar una fotografía de una estantería con libros.
2. Utilizar OpenCV para detectar bordes y contornos de los libros en la imagen.
3. Extraer y guardar imágenes individuales de los libros detectados.
4. Utilizar Tesseract para realizar OCR en las imágenes de los libros y extraer sus títulos.
5. Comparar los títulos extraídos con los libros previamente reconocidos y guardar solo los nuevos títulos reconocidos.
6. Almacenar los títulos reconocidos en un archivo de texto.

Mejoras posibles:
- Mejorar la precisión del reconocimiento de texto utilizando técnicas de preprocesamiento de imágenes.
- Implementar un algoritmo de aprendizaje automático para clasificar los libros por género o tema.

## Instalación

### Requisitos

Este proyecto requiere la instalación de varias bibliotecas y herramientas. Asegúrate de tener instaladas las siguientes bibliotecas antes de intentar ejecutar cualquier proyecto:

```
pip install opencv-python opencv-python-headless mediapipe matplotlib

pip install pytesseract

pip install fuzzywuzzy
```

Además, necesitarás tener instalado [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) en tu sistema. Puedes seguir las instrucciones para su instalación en su página oficial.

### Modelo Darknet

Este proyecto utiliza el modelo [yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) de [darknet](https://github.com/AlexeyAB/darknet.git). Para poder usarlo, necesitarás la [configuración](yolov4.cfg) y las [etiquetas](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names) correspondientes.

## Uso

Para ejecutar cada proyecto, sigue las instrucciones detalladas en los archivos de código fuente individuales. Asegúrate de proporcionar la ruta correcta a tus archivos de vídeo e imágenes en los respectivos proyectos.

### Notas

- El proyecto 1 no está funcionando actualmente.
- El proyecto 2 está en revisión.
- El proyecto 4 está en fase de implementación de nuevas funcionalidades para la gestión de un inventario.