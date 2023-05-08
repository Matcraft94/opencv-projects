# Creado por Lucy
# Fecha: 05/07/2023

import cv2
import os

def capture_images(num_images, output_dir="images"):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Webcam Capture")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Presione la tecla 'c' para capturar una imagen. Presione 'q' para salir.")

    image_count = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            image_path = os.path.join(output_dir, f"image_{image_count:02d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Imagen capturada y guardada en {image_path}")
            image_count += 1

            if image_count >= num_images:
                break

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images(num_images=5)
