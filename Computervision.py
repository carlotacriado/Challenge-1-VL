import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread(r'C:\Users\carlo\OneDrive\Escritorio\trimestre 3\vision & learning\challenge 1\Challenge-1-VL\Challenge-1-VL\Base Images\Frontal\images\067KSH.jpg')

print('image found')

# Redimensionar la imagen
imagen_redimensionada = cv2.resize(imagen, (800, 600))
# Mostrar la imagen
cv2.imshow('Imagen', imagen_redimensionada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen Procesada', imagen_gris)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aplicar GaussianBlur
imagen_blur = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
cv2.imshow('Imagen Blur', imagen_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detección de bordes
bordes = cv2.Canny(imagen_blur, 100, 200)

# Encontrar contornos
contornos, _ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Función para verificar la presencia de azul en una imagen
def contiene_azul(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    azul_pixels = cv2.countNonZero(mask)
    return azul_pixels > 500  # Ajusta este umbral según sea necesario

for contorno in contornos:
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)

    # Si tiene 4 vértices, es un posible rectángulo
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h

        # Filtrar por la relación de aspecto típica de una matrícula
        if 2 < aspect_ratio < 5:
            # Extraer la región de la matrícula
            matricula_recortada = imagen_redimensionada[y:y + h, x:x + w]

            # Comprobar si la región contiene azul
            if contiene_azul(matricula_recortada):
                # Dibujar el rectángulo sobre la matrícula detectada
                cv2.rectangle(imagen_redimensionada, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Redimensionar la matrícula recortada para visualizar
                matricula_recortada = cv2.resize(matricula_recortada, (800, 200))  # Ajusta el tamaño según sea necesario

                # Mostrar la matrícula detectada
                cv2.imshow('Matrícula Detectada', matricula_recortada)

# Mostrar la imagen original con el rectángulo
cv2.imshow('Imagen Original con Matrícula', imagen_redimensionada)
cv2.waitKey(0)

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()


# import cv2
# import os

# # Ruta de la carpeta que contiene las imágenes
# ruta_carpeta = r'C:\Users\carlo\OneDrive\Escritorio\trimestre 3\vision & learning\challenge 1\Challenge-1-VL\Challenge-1-VL\Base Images\Frontal\images'

# # Obtener la lista de archivos en la carpeta
# imagenes = os.listdir(ruta_carpeta)

# # Filtrar solo los archivos de imagen (puedes agregar más extensiones si es necesario)
# extensiones_imagenes = ['.jpg', '.jpeg', '.png']
# imagenes = [img for img in imagenes if os.path.splitext(img)[1].lower() in extensiones_imagenes]

# # Procesar cada imagen
# for nombre_imagen in imagenes:
#     # Crear la ruta completa de la imagen
#     ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)

#     # Cargar la imagen
#     imagen = cv2.imread(ruta_imagen)

#     # Verificar si la imagen se ha cargado correctamente
#     if imagen is None:
#         print(f"Error: No se pudo cargar la imagen {nombre_imagen}.")
#         continue

#     # Mostrar la imagen
#     cv2.imshow('Imagen', imagen)

#     # Esperar a que se presione una tecla para cerrar la ventana
#     cv2.waitKey(0)

# # Cerrar todas las ventanas de OpenCV
# cv2.destroyAllWindows()
