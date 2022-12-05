import numpy as np
import cv2
import os

def Prep(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img, 125, 255,
                       cv2.THRESH_BINARY)
    return np.array(thresh)

def load_digits():
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