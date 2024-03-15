import cv2
import numpy as np


def barrel_distortion(image):
    rows, cols = image.shape[0:2]

    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))
    x_mid = cols / 2.0
    y_mid = rows / 2.0
    xi = xi - x_mid
    yi = yi - y_mid
    r, theta = cv2.cartToPolar(xi / x_mid, yi / y_mid)

    F3 = 0.3
    F5 = 0.12
    r = r + F3 ** 2 + F5 ** 2

    u, v = cv2.polarToCart(r, theta)
    u = u * x_mid + x_mid
    v = v * y_mid + y_mid
    image_barrel = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

    return image_barrel


def pillow_distortion(image):
    rows, cols, channels = image.shape

    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))
    x_mid = cols / 2.0
    y_mid = rows / 2.0
    xi = xi - x_mid
    yi = yi - y_mid
    r, theta = cv2.cartToPolar(xi / x_mid, yi / y_mid)

    F3 = 0.1
    r = r + F3 * r ** 2

    u, v = cv2.polarToCart(r, theta)
    u = u * x_mid + x_mid
    v = v * y_mid + y_mid
    image_pillow = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

    return image_pillow


image = cv2.imread('image.jpg')

image_barrel = barrel_distortion(image)
cv2.imwrite("image_barrel.jpg", image_barrel)

image_unbarrelled = pillow_distortion(image_barrel)
cv2.imwrite("image_unbarrelled.jpg", image_unbarrelled)

image_pillow = pillow_distortion(image)
cv2.imwrite("image_pillow.jpg", image_pillow)
image_unpillowed = barrel_distortion(image_pillow)
cv2.imwrite("image_unpillowed.jpg", image_unpillowed)



