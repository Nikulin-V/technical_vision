import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

hist_size = 256
hist_range = (0, 256)


# Отображает гистограммы и сохраняет их в файл
def show_hists(red_hist, green_hist, blue_hist, title="RGB Histograms", filename=None):
    if filename is None:
        filename = title

    x = range(*hist_range)
    plt.title(title)
    for hist, color in [(red_hist, "red"), (green_hist, "green"), (blue_hist, "blue")]:
        y = list(map(lambda value: value[0], hist))
        plt.plot(x, y, color=color)
    figure = plt.gcf()
    plt.show()
    plt.draw()
    figure.savefig(filename + ".jpg")


# Создаёт гистограммы и отображает их
def show_and_save_image_hists(img, title="RGB Histograms", filename="hist_image"):
    image_bgr = cv2.split(img)
    b_hist = cv2.calcHist(image_bgr, [0], None, [hist_size], hist_range)
    g_hist = cv2.calcHist(image_bgr, [1], None, [hist_size], hist_range)
    r_hist = cv2.calcHist(image_bgr, [2], None, [hist_size], hist_range)
    show_hists(r_hist, g_hist, b_hist, title, filename)


# Обрабатывает изображение с помощью нелинейного растяжения динамического диапазона
def stretch_image_nonlinear(img, alpha=0.5):
    if img.dtype == np.uint8:
        img_new = img.astype(np.float32) / 255
    else:
        img_new = img

    # Разбиваем изображение на слои
    img_bgr = cv2.split(img_new)
    img_new_bgr = []
    for layer in img_bgr:
        img_min = layer.min()
        img_max = layer.max()
        # Формула нелинейного растяжения динамического диапазона
        img_new = np.clip((((layer - img_min) / (img_max - img_min)) ** alpha), 0, 1)
        img_new_bgr.append(img_new)

    # Собираем слои в изображение
    img_new = cv2.merge(img_new_bgr)
    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)
    return img_new


# Выравнивает гистограммы канала методом равномерного преобразования
def equalize_channel_uniform(ch):
    img_min = ch.min()
    img_max = ch.max()
    histogram = cv2.calcHist([ch], [0], None, [hist_size], hist_range)
    # Вычисляем кумулятивную гистограмму
    histogram_cumulative = np.cumsum(histogram) / (rows * cols)
    # Создаём LUT
    lookup_table = np.uint8((img_max - img_min) * histogram_cumulative + img_min)
    # Применяем LUT
    ch_equalized = cv2.LUT(ch, lookup_table)
    return ch_equalized


# Выравнивает гистограммы изображения методом экспоненциального преобразования
def equalize_image_exponential(img, alpha=0.5):
    rows, cols, _ = img.shape
    img_min = img.min()

    # Вычисляем кумулятивную гистограмму
    histogram = cv2.calcHist([img], [0], None, [hist_size], hist_range)
    histogram_cumulative = np.cumsum(histogram) / (rows * cols)

    # Разбиваем изображение на каналы
    channels = cv2.split(img)
    channels_equalized = []
    for ch in channels:
        ch_equalized = np.zeros_like(ch, dtype=np.float32)
        # Обрабатываем каждый пиксель канала
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                pixel = ch[i, j]
                # Защита от деления на 0
                x = 1 - histogram_cumulative[pixel] or 0.001
                # Формула экспоненциального преобразования
                ch_equalized[i][j] = img_min - ((1 / alpha) * math.log(x))
        # Нормализация канала
        ch_equalized = cv2.normalize(ch_equalized, None, 0, 255, cv2.NORM_MINMAX)
        ch_equalized = np.clip(ch_equalized, 0, 255).astype(np.uint8)
        channels_equalized.append(ch_equalized.astype(np.uint8))

    # Собираем каналы в изображение
    img_new = cv2.merge(channels_equalized)
    return img_new


# Выравнивает гистограммы изображения преобразованием по закону Рэлея
def equalize_image_rayleigh_law(img, alpha=0.5):
    rows, cols, _ = img.shape
    img_min = img.min()

    # Вычисляем кумулятивную гистограмму
    histogram = cv2.calcHist([img], [0], None, [hist_size], hist_range)
    histogram_cumulative = np.cumsum(histogram) / (rows * cols)

    # Разбиваем изображение на каналы
    channels = cv2.split(img)
    channels_equalized = []
    for ch in channels:
        ch_equalized = np.zeros_like(ch, dtype=np.float32)
        # Обрабатываем каждый пиксель канала
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                pixel = ch[i, j]
                # Защита от деления на 0
                x = 1 - histogram_cumulative[pixel] or 0.001
                # Формула преобразования по закону Рэлея
                ch_equalized[i][j] = img_min + math.sqrt(2 * alpha ** 2 * math.log(1 / x))
        # Нормализация канала
        ch_equalized = cv2.normalize(ch_equalized, None, 0, 255, cv2.NORM_MINMAX)
        ch_equalized = np.clip(ch_equalized, 0, 255).astype(np.uint8)
        channels_equalized.append(ch_equalized.astype(np.uint8))

    # Собираем каналы в изображение
    img_new = cv2.merge(channels_equalized)
    return img_new


# Выравнивает гистограммы изображения преобразованием по закону степени 2/3
def equalize_image_power_2_3_law(img):
    rows, cols, _ = img.shape

    # Вычисляем кумулятивную гистограмму
    histogram = cv2.calcHist([img], [0], None, [hist_size], hist_range)
    histogram_cumulative = np.cumsum(histogram) / (rows * cols)

    # Разбиваем изображение на каналы
    channels = cv2.split(img)
    channels_equalized = []
    for ch in channels:
        ch_equalized = np.zeros_like(ch, dtype=np.float32)
        # Обрабатываем каждый пиксель канала
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                # Формула преобразования по закону степени 2/3
                ch_equalized[i][j] = histogram_cumulative[ch[i, j]] ** (2 / 3)
        # Нормализация канала
        ch_equalized = cv2.normalize(ch_equalized, None, 0, 255, cv2.NORM_MINMAX)
        ch_equalized = np.clip(ch_equalized, 0, 255).astype(np.uint8)
        channels_equalized.append(ch_equalized.astype(np.uint8))

    # Собираем каналы в изображение
    img_new = cv2.merge(channels_equalized)
    return img_new

# Выравнивает гистограммы изображения методом гиперболического преобразования
def equalize_image_hyperbolic(img, alpha=0.5):
    rows, cols, _ = img.shape

    # Вычисляем кумулятивную гистограмму
    histogram = cv2.calcHist([img], [0], None, [hist_size], hist_range)
    histogram_cumulative = np.cumsum(histogram) / (rows * cols)

    # Разбиваем изображение на каналы
    channels = cv2.split(img)
    channels_equalized = []
    for ch in channels:
        ch_equalized = np.zeros_like(ch, dtype=np.float32)
        # Обрабатываем каждый пиксель канала
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                # Формула преобразования по закону степени 2/3
                ch_equalized[i][j] = alpha ** histogram_cumulative[ch[i, j]]
        # Нормализация канала
        ch_equalized = cv2.normalize(ch_equalized, None, 0, 255, cv2.NORM_MINMAX)
        ch_equalized = np.clip(ch_equalized, 0, 255).astype(np.uint8)
        channels_equalized.append(ch_equalized.astype(np.uint8))

    # Собираем каналы в изображение
    img_new = cv2.merge(channels_equalized)
    return img_new

# Считываем изображение
image = cv2.imread("low_contrast_image.jpg")
rows, cols, _ = image.shape
histogram_size = image.shape
# Разбиваем изображение на каналы
channels = cv2.split(image)
# Сохраняем исходные гистограммы
title = "Original Histograms"
show_and_save_image_hists(image, title)
histogram = cv2.imread(title + ".jpg")

# Нелинейное растяжение динамического диапазона
image_stretched_nonlinear = stretch_image_nonlinear(image, alpha=0.5)
cv2.imwrite("image_stretched_nonlinear.jpg", image_stretched_nonlinear)
title = "Stretched Nonlinear Histograms"
show_and_save_image_hists(image_stretched_nonlinear, title)

# Равномерное преобразование
image_equalized_uniform = cv2.merge([equalize_channel_uniform(ch) for ch in channels])
cv2.imwrite("image_equalized_uniform.jpg", image_equalized_uniform)
title = "Equalized Uniform Histograms"
show_and_save_image_hists(image_equalized_uniform, title)

# Экспоненциальное преобразование
image_equalized_exponential = equalize_image_exponential(image, alpha=0.5)
cv2.imwrite("image_equalized_exponential.jpg", image_equalized_exponential)
title = "Equalized Exponential Histograms"
show_and_save_image_hists(image_equalized_exponential, title)

# Преобразование по закону Рэлея
image_equalized_rayleigh_law = equalize_image_rayleigh_law(image, alpha=0.5)
cv2.imwrite("image_equalized_rayleigh_law.jpg", image_equalized_rayleigh_law)
title = "Equalized Rayleigh Law Histograms"
show_and_save_image_hists(image_equalized_rayleigh_law, title)

# Преобразование по закону степени 2/3
image_equalized_power_2_3_law = equalize_image_power_2_3_law(image)
cv2.imwrite("image_equalized_power_2_3_law.jpg", image_equalized_power_2_3_law)
title = "Equalized Power 2/3 Law Histograms"
filename = "Equalized Power 2_3 Law Histograms"
show_and_save_image_hists(image_equalized_power_2_3_law, title, filename)

# Гиперболическое преобразование
image_equalized_hyperbolic = equalize_image_hyperbolic(image, alpha=0.5)
cv2.imwrite("image_equalized_hyperbolic.jpg", image_equalized_hyperbolic)
title = "Equalized Hyperbolic Histograms"
show_and_save_image_hists(image_equalized_hyperbolic, title)

# Склеиваем результирующие изображения для сравнения
result = np.vstack((image, image_stretched_nonlinear, image_equalized_uniform, image_equalized_exponential,
                    image_equalized_rayleigh_law, image_equalized_power_2_3_law, image_equalized_hyperbolic))
cv2.imwrite("result.jpg", result)
