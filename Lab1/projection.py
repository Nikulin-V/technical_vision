import cv2
import numpy as np
import matplotlib.pyplot as plt

# Считываем изображение
image = cv2.imread('objects.png')

# Преобразовываем к чёрно-белому
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Вычисляем проекции изображений
projection_ox = np.sum(gray_image, axis=0)
projection_oy = np.sum(gray_image, axis=1)
projection_ox = (projection_ox - np.min(projection_ox)) / (np.max(projection_ox) - np.min(projection_ox))
projection_oy = (projection_oy - np.min(projection_oy)) / (np.max(projection_oy) - np.min(projection_oy))

# Определение положения объектов
threshold = 0.75
thresholded_ox = (projection_ox > threshold).astype(np.uint8)
thresholded_oy = (projection_oy > threshold).astype(np.uint8)

# Отрисовка исходного изображения
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Отрисовка горизонтальной проекции
plt.title('Horizontal Projection')
plt.plot(range(len(projection_ox)), projection_ox)
plt.show()

# Отрисовка вертикальной проекции
plt.title('Vertical Projection')
plt.plot(projection_oy[::-1], range(len(projection_oy)))
plt.show()

# Отрисовка горизонтальных областей предполагаемого местоположения объектов
plt.title('Horizontal Areas')
plt.plot(range(len(thresholded_ox)), thresholded_ox, color='black')
plt.show()

# Отрисовка вертикальных областей предполагаемого местоположения объектов
plt.title('Vertical Areas')
plt.plot(thresholded_oy[::-1], range(len(thresholded_oy)), color='black')
plt.show()
