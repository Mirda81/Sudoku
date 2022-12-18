import cv2
import os
import numpy as np


def Prep(img):
    """
    :param img: image of number
    :return: binary image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 125, 255,
                                cv2.THRESH_BINARY)
    return np.array(thresh)


def load_digits():
    """
    loading digits from folder for model training
    """
    Slozka = r'Digits/'
    Kategorie = [str(cislo) for cislo in range(10)]
    data2 = []
    for category in Kategorie:
        folder = os.path.join(Slozka, category)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            label = int(Kategorie.index(category))
            arr = cv2.imread(img_path)
            new_arr = cv2.resize(arr, (40, 40))
            new_arr = Prep(new_arr)
            data2.append([new_arr, label])
    return data2


def load_sudoku_images(file):
    """
    loading test images for app_photo
    """
    data = []

    for img in os.listdir(file):
        img_path = os.path.join(file, img)
        arr = cv2.imread(img_path)
        new_arr = cv2.resize(arr, (540, 540), interpolation=cv2.INTER_LINEAR)
        data.append(new_arr)
    data = np.array(data)
    return data

def image_overlay_second_method(img1, img2, location, min_thresh=0, is_transparent=False):
    h, w = img1.shape[:2]
    h1, w1 = img2.shape[:2]
    x, y = location
    roi = img1[y:y + h1, x:x + w1]

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img_bg, img_fg)
    if is_transparent:
        dst = cv2.addWeighted(img1[y:y + h1, x:x + w1], 0.1, dst, 0.9, None)
    img1[y:y + h1, x:x + w1] = dst
    return img1