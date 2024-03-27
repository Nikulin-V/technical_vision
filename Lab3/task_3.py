import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')


def apply_gaussian_noise(image):
    mean = 3
    var = 0.1
    noisy_image = image.copy()
    rows, cols, channels = noisy_image.shape
    gauss = np.random.normal(mean, var, (rows, cols, channels)).astype(np.float32)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image


new_image = apply_gaussian_noise(image)


# Медианная фильтрация
def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


# Взвешенная медианная фильтрация

def weighted_rank_filter(image, k_size=(3, 3), rank=4):
    kernel = np.ones(k_size, dtype=np.float32)
    rows, cols = image.shape[0:2]

    # Преобразование изображения в формат float и добавление граничных пикселей
    if image.dtype == np.uint8:
        image_copy = image.astype(np.float32) / 255
    else:
        image_copy = image.copy()

    image_copy = cv2.copyMakeBorder(image_copy,
                                    int(k_size[0] - 1), int(k_size[0] / 2),
                                    int(k_size[1] - 1), int(k_size[1] / 2),
                                    cv2.BORDER_REPLICATE)

    # Создание массивов для каждого элемента ядра
    image_layers = np.zeros(image.shape + (k_size[0] * k_size[1],), dtype=np.float32)

    if image.ndim == 2:
        for i in range(k_size[0]):
            for j in range(k_size[1]):
                image_layers[:, :, i * k_size[1] + j] = kernel[i, j] * image_copy[i:i + rows, j:j + cols]
        # Сортировка массивов
        image_layers.sort()
        # Выбор слоя с рангом
        filtered_image = image_layers[:, :, rank]
    else:
        for i in range(k_size[0]):
            for j in range(k_size[1]):
                image_layers[:, :, :, i * k_size[1] + j] = kernel[i, j] * image_copy[i:i + rows, j:j + cols, :]
        # Сортировка массивов
        image_layers.sort()
        # Выбор слоя с рангом
        filtered_image = image_layers[:, :, :, rank]
    # Преобразование обратно в uint8 при необходимости
    if image.dtype == np.uint8:
        filtered_image = (255 * filtered_image).clip(0, 255).astype(np.uint8)
    return filtered_image


# Винеровская фильтрация
def wiener_filter(image, kernel_size=(7, 7)):
    kernel = np.ones((kernel_size[0], kernel_size[1]))
    rows, cols = image.shape[:2]
    if image.dtype == np.uint8:
        img_copy = image.astype(np.float32) / 255
    else:
        img_copy = image.copy()
    img_copy = cv2.copyMakeBorder(img_copy,
                                  int((kernel_size[0] - 1) / 2), int(kernel_size[0] / 2),
                                  int((kernel_size[1] - 1) / 2), int(kernel_size[1] / 2),
                                  cv2.BORDER_REPLICATE)
    bgr_planes = cv2.split(img_copy)
    bgr_planes_2 = []

    for plane in bgr_planes:
        plane_power = np.power(plane, 2)
        m = np.zeros(image.shape[:2], np.float32)
        q = np.zeros(image.shape[:2], np.float32)

        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                m += kernel[i, j] * plane[i:i + rows, j:j + cols]
                q += np.power(kernel, 2)[i, j] * plane_power[i:i + rows, j:j + cols]
        m /= np.sum(kernel)
        q /= np.sum(kernel)
        plane_2 = plane[(kernel_size[0] - 1) // 2: (kernel_size[0] - 1) // 2 + rows,
                  (kernel_size[1] - 1) // 2: (kernel_size[1] - 1) // 2 + cols]
        v = np.sum(q) / image.size
        valid_q = np.where(q != 0, q, 1)  # Заменяем нулевые значения в q на 1
        # Избегаем деления на ноль и умножения на недопустимые значения
        plane_2 = np.where(q < v, m, (plane_2 - m) * (1 - v / valid_q) + m)
        bgr_planes_2.append(plane_2)
    filtered_image = cv2.merge(bgr_planes_2)
    if image.dtype == np.uint8:
        filtered_image = (255 * filtered_image).clip(0, 255).astype(np.uint8)
    return filtered_image


# Адаптивная медианная фильтрация
def adaptive_median_filter(image, window_size, max_window_size):
    height, width = image.shape[:2]
    filtered_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            window_size_curr = window_size
            while window_size_curr <= max_window_size:
                window = image[max(0, y - window_size_curr // 2):min(height, y + window_size_curr // 2 + 1),
                         max(0, x - window_size_curr // 2):min(width, x + window_size_curr // 2 + 1)]
                median = np.median(window)
                min_val = np.min(window)
                max_val = np.max(window)

                if min_val < median < max_val:
                    filtered_image[y, x] = np.clip(image[y, x], min_val, max_val)
                    break
                else:
                    window_size_curr += 2
            if window_size_curr > max_window_size:
                filtered_image[y, x] = image[y, x]
    return filtered_image


median_image = median_filter(new_image, 5)
median_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('median_image.jpg', median_image)

weighted_rank_image = weighted_rank_filter(new_image, (3, 3), 4)
weighted_rank_image = cv2.cvtColor(weighted_rank_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('weighted_rank_image.jpg', weighted_rank_image)

wiener_image = wiener_filter(new_image, (3, 3))
wiener_image = cv2.cvtColor(wiener_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('wiener_image.jpg', wiener_image)

adapt_image = adaptive_median_filter(new_image, 3, 11)
adapt_image = cv2.cvtColor(adapt_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('adapt_image.jpg', adapt_image)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(median_image)
axs[0, 0].set_title('Медианная фильтрация')
axs[0, 0].axis('off')

axs[0, 1].imshow(weighted_rank_image)
axs[0, 1].set_title('Взвешенная фильтрация')
axs[0, 1].axis('off')

axs[1, 0].imshow(wiener_image)
axs[1, 0].set_title('Винеровская фильтрация')
axs[1, 0].axis('off')

axs[1, 1].imshow(adapt_image)
axs[1, 1].set_title('Адаптивная фильтрация')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
