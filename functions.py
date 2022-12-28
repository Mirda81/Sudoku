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
                                cv2.THRESH_BINARY_INV)
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

def camera_set(width, height):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # change brightness to 150
    cap.set(10, 150)
    return cap