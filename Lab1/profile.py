import cv2
import matplotlib.pyplot as plt


# Выводит график профиля
def show_profile(profile_list):
    x = range(len(profile_list))
    y = profile_list
    plt.title("Profile")
    plt.plot(x, y, color="black")
    plt.show()


# Считываем штрих-код
image = cv2.imread("barcode.png", cv2.IMREAD_COLOR)
# Берём ряд пикселей по середине изображения
profile = image[round(image.shape[0] / 2), :]
# Выводим профиль
show_profile(profile)
