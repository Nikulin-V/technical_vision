import cv2
import numpy as np
import skimage


# Задание 1
# Импульсный шум
def impulse_noise(img):
    # Параметры
    d = 0.05
    s_vs_p = 0.5

    # Создаём рандомные числа
    rng = np.random.default_rng()
    vals = rng.random(img.shape)

    # Соль
    I_out = np.copy(img)
    if I_out.dtype == np.uint8:
        I_out[vals < d * s_vs_p] = 255
    else:
        I_out[vals < d * s_vs_p] = 1.0

    # Перец
    I_out[np.logical_and(vals >= d * s_vs_p, vals < d)] = 0

    return I_out


# Мультипликативный шум
def multiplicative_noise(img):
    # Параметр
    var = 0.05

    # Генерация рандомных чисел
    rng = np.random.default_rng()
    gauss = rng.normal(0, var ** 0.5, img.shape)

    # Обработка uchar и float изображений раздельно
    if img.dtype == np.uint8:
        I_f = img.astype(np.float32)
        I_out = (I_f + I_f * gauss). \
            clip(0, 255).astype(np.uint8)
    else:
        I_out = img + img * gauss
    return I_out


# Гауссовский шум
def gauss_noise(img):
    # Параметры
    mean = 0
    var = 0.01

    # Генерация рандомных чисел
    rng = np.random.default_rng()
    gauss = rng.normal(mean, var ** 0.5, img.shape)
    gauss = gauss.reshape(img.shape)

    # Обработка uchar и float изображений раздельно
    if img.dtype == np.uint8:
        I_out = (img.astype(np.float32) +
                 gauss * 255).clip(0, 255).astype(np.uint8)
    else:
        I_out = (img + gauss).astype(np.float32)
    return I_out


# Шум квантования
def quant_noise(img):
    rng = np.random.default_rng()
    if img.dtype == np.uint8:
        I_f = img.astype(np.float32) / 255
        vals = len(np.unique(I_f))
        vals = 2 ** np.ceil(np.log2(vals))
        I_out = (255 * (rng.poisson(I_f * vals) / float(vals)).clip(0, 1)).astype(np.uint8)
    else:
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        I_out = \
            rng.poisson(img * vals) / float(vals)
    return I_out


# Задание 2
# Функция для применения фильтра Гаусса
def gaussian_filter(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


# Контргармонический усредняющий фильтр
def contraharmonic_mean_filter(img, q):
    kernel = np.ones((3, 3), np.float32) / (q + 1)
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img


def task2():
    median_image = cv2.imread('median_image.jpg')
    weighted_rank_image = cv2.imread('weighted_rank_image.jpg')
    wiener_image = cv2.imread('wiener_image.jpg')
    adapt_image = cv2.imread('adapt_image.jpg')
    broken_images = [median_image, weighted_rank_image, wiener_image, adapt_image]

    for img in broken_images:
        for sigma in [1, 2, 3, 4]:
            filtered_img = gaussian_filter(img.copy(), sigma)
            cv2.imwrite(f'gaussian_filter_image_sigma_{sigma}', filtered_img)
        for q in [2, 4, 8, 16]:
            filtered_img = contraharmonic_mean_filter(img.copy(), q)
            cv2.imwrite(f'contraharmonic_mean_filter_image_q_{q}', filtered_img)


# Задание 3


# Задание 4
def rob(image):
    image = image.astype(np.float32)
    G_x = np.array([[1, -1], [0, 0]])
    G_y = np.array([[1, 0], [-1, 0]])
    I_x = cv2.filter2D(image, -1, G_x)
    I_y = cv2.filter2D(image, -1, G_y)
    return cv2.magnitude(I_x, I_y)


def previtt(image):
    image = image.astype(np.float32)
    G_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    G_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    I_x = cv2.filter2D(image, -1, G_x)
    I_y = cv2.filter2D(image, -1, G_y)
    return cv2.magnitude(I_x, I_y)


def sob(image):
    image = image.astype(np.float32)
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    I_x = cv2.filter2D(image, -1, G_x)
    I_y = cv2.filter2D(image, -1, G_y)
    return cv2.magnitude(I_x, I_y)


# Фильтр Лапласа
def laplacian(image):
    image = image.astype(np.float32)
    return cv2.Laplacian(image, cv2.CV_32F)


# Алгоритм Кэнни
def canny(image, t1, t2):
    image = image.astype(np.uint8)
    return cv2.Canny(image, t1, t2)


image = cv2.imread('image.jpg')
cv2.imwrite('impulse_noise_image.jpg', impulse_noise(image))
cv2.imwrite('multiplicative_noise_image.jpg', multiplicative_noise(image))
cv2.imwrite('gauss_noise_image.jpg', gauss_noise(image))
cv2.imwrite('quant_noise_image.jpg', quant_noise(image))

cv2.imwrite('rob_image.jpg', rob(image))
cv2.imwrite('previtt_image.jpg', previtt(image))
cv2.imwrite('sob_image.jpg', sob(image))
cv2.imwrite('laplacian_image.jpg', laplacian(image))
cv2.imwrite('canny_image.jpg', canny(image, 0, 800))
