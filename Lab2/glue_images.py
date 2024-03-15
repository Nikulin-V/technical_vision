import cv2
import numpy as np


def cut_image_to_parts(image, part_prefix='part'):
    # Получаем размеры изображения
    rows, cols = image.shape[0:2]

    # Количество частей изображения
    cols_count = 1
    rows_count = 2

    # Вычисление размеров частей изображения
    part_height = rows // rows_count
    part_width = cols // cols_count

    tiles = []
    for i in range(rows_count):
        for j in range(cols_count):
            # Вычисление координат части изображения
            x_offset = j * part_width
            y_offset = i * part_height
            # Получение части изображения
            part = image[y_offset:y_offset + part_height, x_offset:x_offset + part_width, :]
            # Добавление части в список
            tiles.append(part)

    # Сохранение
    for i, part in enumerate(tiles):
        cv2.imwrite(f'{part_prefix}_{i+1}.jpg', part)


def glue_images(part_prefix='part'):
    top_part = cv2.imread(f'{part_prefix}_1.jpg')
    bottom_part = cv2.imread(f'{part_prefix}_2.jpg')
    template = top_part[:, :, :]
    res = cv2.matchTemplate(bottom_part, template, cv2.TM_CCOEFF)

    # Определение точки соединения
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    y_offset = max_loc[1]

    # Выравнивание и объединение изображений
    bottom_part = bottom_part[y_offset:, :, :]
    image_glued = np.vstack((top_part, bottom_part))

    # Отображение результата
    cv2.imwrite('glued_image.jpg', image_glued)


image = cv2.imread("image.jpg")
cut_image_to_parts(image, part_prefix="nature")
glue_images(part_prefix="nature")
