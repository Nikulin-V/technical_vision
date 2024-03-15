# Считываем изображение
import math

import cv2
import numpy as np


def shift_image(image, x=50, y=100):
    """Сдвигает изображение"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([
        [1, 0, x],
        [0, 1, y]
    ])
    # Применяем матрицу
    image_shifted = cv2.warpAffine(image, T, (cols, rows))
    return image_shifted


def reflect_image_ox(image):
    """Отражает изображение по оси OX"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([
        [1, 0, 0],
        [0, -1, rows - 1]
    ])
    # Применяем матрицу
    image_reflected_ox = cv2.warpAffine(image, T, (cols, rows))
    # или image_reflected_ox = cv2.flip(image, 0)
    return image_reflected_ox


def uniform_image_scale(image, scale_x=2, scale_y=2):
    """Масштабирует изображение"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([
        [scale_x, 0, 0],
        [0, scale_y, 0]
    ])
    # Применяем матрицу
    image_scaled = cv2.warpAffine(image, T, (int(cols * scale_x), int(rows * scale_y)))
    # или image_scaled = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    return image_scaled


def rotate_image_upper_left_corner(image, angle_degrees=17):
    """Поворачивает изображение относительно верхнего левого угла"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    phi = angle_degrees * math.pi / 180
    # Составляем матрицу
    T = np.float32([
        [math.cos(phi), -math.sin(phi), 0],
        [math.sin(phi), math.cos(phi), 0]
    ])
    # Применяем матрицу
    image_rotated_upper_left_corner = cv2.warpAffine(image, T, (cols, rows))
    return image_rotated_upper_left_corner


def rotate_image(image, angle_degrees=17, point=None):
    """Поворачивает изображение"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    if point is None:
        point = ((cols - 1) / 2, (rows - 1) / 2)
    phi = angle_degrees * math.pi / 180
    point_x, point_y = point
    # Составляем матрицу
    T1 = np.float32(
        [[1, 0, -point_x],
         [0, 1, -point_y],
         [0, 0, 1]])
    T2 = np.float32(
        [[math.cos(phi), - math.sin(phi), 0],
         [math.sin(phi), math.cos(phi), 0],
         [0, 0, 1]])
    T3 = np.float32(
        [[1, 0, point_x],
         [0, 1, point_y],
         [0, 0, 1]])
    T = np.matmul(T3, np.matmul(T2, T1))[0:2, :]
    # или T = cv2.getRotationMatrix2D(([point_x, point_y], phi, 1))
    # Применяем матрицу
    image_rotated = cv2.warpAffine(image, T, (cols, rows))
    return image_rotated


def bevel_image(image, s=0.3):
    """Скашивает изображение"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([
        [1, s, 0],
        [0, 1, 0]
    ])
    # Применяем матрицу
    image_beveled = cv2.warpAffine(image, T, (cols, rows))
    return image_beveled


def piece_wise_linear_stretch(image, stretch=2):
    """Растягивает правую половину изображения по оси OX"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([[stretch, 0, 0], [0, 1, 0]])
    image_stretched_piece_wise = image.copy()
    # Применяем матрицу
    image_stretched_piece_wise[:, int(cols / 2):, :] = cv2.warpAffine(
        image_stretched_piece_wise[:, int(cols / 2):, :], T, (cols - int(cols / 2), rows)
    )
    return image_stretched_piece_wise


def projective_transform_image(image, a=1.1, b=0.35, c=0, d=0.2, e=1.1, f=0, g=0.00075, h=0.0005, i=1):
    """Проекционное преобразование"""
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.float32([
        [a, b, c],
        [d, e, f],
        [g, h, i]
    ])
    # Применяем матрицу
    image_transformed = cv2.warpPerspective(image, T, (cols, rows))
    return image_transformed


def polynomial_transform_image(image):
    """Полиномиальное преобразование"""
    rows, cols = image.shape[0:2]
    # Составляем матрицу
    T = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.00001, 0],
        [0.002, 0],
        [0.001, 0]
    ])
    image_polynomial = np.zeros(image.shape, image.dtype)
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Вычисляем новые координаты X и Y
    x_new = np.round(T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]).astype(
        np.float32)
    y_new = np.round(T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]).astype(
        np.float32)
    mask = np.logical_and(np.logical_and(x_new >= 0, x_new < cols), np.logical_and(y_new >= 0, y_new < rows))

    # Применяем маску
    if image.ndim == 2:
        image_polynomial[y_new[mask].astype(int), x_new[mask].astype(int)] = image[y[mask], x[mask]]
    else:
        image_polynomial[y_new[mask].astype(int), x_new[mask].astype(int), :] = image[y[mask], x[mask], :]

    return image_polynomial


def sinusoid_transform_image(image):
    """Синусоидальное преобразование"""
    rows, cols = image.shape[0:2]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u + 20 * np.sin(2 * math.pi * v / 90)
    image_transformed = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    return image_transformed



transformations = {
    'shift': shift_image,
    'reflect': reflect_image_ox,
    'uniform_scale': uniform_image_scale,
    'rotate_left_upper_corner': rotate_image_upper_left_corner,
    'rotate': rotate_image,
    'bevel': bevel_image,
    'piece_wise_linear_stretch': piece_wise_linear_stretch,
    'projective_transform': projective_transform_image,
    'polynomial_transform': polynomial_transform_image,
    'sinusoid_transform': sinusoid_transform_image
}

image = cv2.imread('image.jpg')
for name, transformation in transformations.items():
    image_result = transformation(image)
    cv2.imwrite(name + ".jpg", image_result)
