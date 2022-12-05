import cv2
import numpy as np

def Preprocess(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray

def extract_frame(img):
    ramecek = np.zeros((img.shape), np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 50))
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(img) / (close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    max = 0
    hit = False
    for kontura in contours:
        obsah = cv2.contourArea(kontura)
        peri = cv2.arcLength(kontura, True)
        vektory = cv2.approxPolyDP(kontura, 0.02 * peri, True)
        if (len(vektory == 4)) and (obsah > max):
            max = obsah
            biggest_contour = vektory
            hit = True
    hit

    cv2.drawContours(ramecek, [biggest_contour], 0, 255, -1)
    cv2.drawContours(ramecek, [biggest_contour], 0, 0, 2)
    res = cv2.bitwise_and(res, ramecek)

    return res, biggest_contour

def Perspective_transform(img, shape, kontura):
    biggest_contour = kontura.reshape(len(kontura), 2)
    suma_vekt = biggest_contour.sum(1)
    print(suma_vekt)
    suma_vekt2 = np.delete(biggest_contour, [np.argmax(suma_vekt), np.argmin(suma_vekt)], 0)
    suma_vekt2[:, 0]

    corners = np.float32([biggest_contour[np.argmin(suma_vekt)], suma_vekt2[np.argmax(suma_vekt2[:, 0])],
                          suma_vekt2[np.argmin(suma_vekt2[:, 0])], biggest_contour[np.argmax(suma_vekt)]])

    pts2 = np.float32([[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(corners, pts2)
    result = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))

    return result

def frame_kont(img):

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 50))
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(img) / (close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))


    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    max = 0
    hit = False
    for kontura in contours:
        obsah = cv2.contourArea(kontura)
        peri = cv2.arcLength(kontura, True)
        vektory = cv2.approxPolyDP(kontura, 0.02 * peri, True)
        if (len(vektory == 4)) and (obsah > max):
            max = obsah
            biggest_contour = kontura
            hit = True

    return biggest_contour
