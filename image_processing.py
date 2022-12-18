import cv2
import numpy as np
import keras
import pickle


def Preprocess(img):
    """
    :param img: input image
    :return: blurred gray image
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray


def extract_frame(img):
    """
    :param img: input image
    :return: image with extracted sudoku grid, biggest contour
    """
    ramecek = np.zeros((img.shape), np.uint8)
    res2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 9, 5)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = []
    res = []
    max = 0
    for kontura in contours:
        obsah = cv2.contourArea(kontura)
        peri = cv2.arcLength(kontura, True)
        vektory = cv2.approxPolyDP(kontura, 0.01 * peri, True)
        if (len(vektory) == 4) and (obsah > max) and (obsah > 10000):
            max = obsah
            biggest_contour = vektory
    if len(biggest_contour) > 0:
        cv2.drawContours(ramecek, [biggest_contour], 0, 255, -1)
        cv2.drawContours(ramecek, [biggest_contour], 0, 0, 2)
        res = cv2.bitwise_and(img, ramecek)

    return res, biggest_contour, ramecek


def get_corners(contour):
    """
    :param contour: contour of sudoku grid - list
    :return: sorted corners coordination - list
    """
    biggest_contour = contour.reshape(len(contour), 2)
    suma_vekt = biggest_contour.sum(1)
    suma_vekt2 = np.delete(biggest_contour, [np.argmax(suma_vekt), np.argmin(suma_vekt)], 0)

    corners = np.float32([biggest_contour[np.argmin(suma_vekt)], suma_vekt2[np.argmax(suma_vekt2[:, 0])],
                          suma_vekt2[np.argmin(suma_vekt2[:, 0])], biggest_contour[np.argmax(suma_vekt)]])

    return corners


def Perspective_transform(img, shape, corners):
    """
    :param img: input image - numPy array
    :param shape: shape of returned image - tuple (w,h)
    :param corners: list of corners coordinations
    :return: perspective transformed image - numPy array
    """
    pts2 = np.float32(
        [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])  # Apply Perspective Transform Algorithm

    matrix = cv2.getPerspectiveTransform(corners, pts2)
    result = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))

    return result


def extract_numbers(img):
    """
    :param img: input binary image
    :return: image with extracted numbers, list of countours stats(left, top, width, height, area), centroid coordinations
    """
    result = preProcess_numbers(img)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(result)
    viz = np.zeros_like(result, np.uint8)
    centroidy = []
    stats_numbers = []

    for i, stat in enumerate(stats):
        if i == 0:
            continue
        if stat[4] > 50 and stat[2] > 5 and stat[3] > 5 and stat[3] < 40 and stat[2] < 40 and stat[0] > 0 and stat[
            1] > 0 and stat[3] / stat[2] > 1 and stat[3] / stat[2] < 5:
            viz[labels == i] = 255
            centroidy.append(centroids[i])
            stats_numbers.append(stat)

    stats_numbers = np.array(stats_numbers)
    centroidy = np.array(centroidy)
    return viz, stats_numbers, centroidy


def preProcess_numbers(img):
    """
    :param img: image of number
    :return: processed image
    """
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img


def center_numbers(img, stats, centroids):
    """
    :param img: image with extracted numbers
    :param stats: stats of contours
    :param centroids: centroids of conours
    :return: image with centered number to grid
    """
    centered_num_grid = np.zeros_like(img, np.uint8)
    for i, number in enumerate(stats):
        cropped_num = img[number[1]:number[1] + number[3], number[0]:number[0] + number[2]]
        center = centroids[i]
        offset_x = int(np.round(np.round(center[0] / 25, 0) * 25 - center[0], 0))
        offset_y = int(np.round(np.round(center[1] / 25, 0) * 25 - center[1], 0))
        centered_num_grid[number[1] + offset_y:number[1] + number[3] + offset_y,
        number[0] + offset_x:number[0] + number[2] + offset_x] = img[number[1]:number[1] + number[3],
                                                                 number[0]:number[0] + number[2]]
    return centered_num_grid


def proccesCell(img):
    """
    :param img: image of specific cell with number
    :return: binary iversion image, cropped
    """
    kernel = np.ones((3, 3), np.uint8)
    # closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    ret, thresh = cv2.threshold(img, 125, 255,
                                cv2.THRESH_BINARY_INV)
    cropped_img = thresh[5:thresh.shape[0] - 5, 5:thresh.shape[0] - 5]
    resized = cv2.resize(cropped_img, (40, 40))
    return resized


def predict_numbers(numbers, matice, model):
    """
    :param numbers: image with extracted numbers
    :param matice: empty matrix
    :param model: model for prediction
    :return: matrix with predicted numbers, empty cells = 0
    """
    y = 0
    step = 50
    while y < 450:
        x = 0
        while x < 450:
            vysek = numbers[y:y + 50, x:x + 50]
            if np.sum(vysek) == 0:
                predikce = 0
            else:
                vysek = proccesCell(vysek)
                vysek = vysek / 255
                predikce = np.argmax(model.predict(vysek.reshape(1, 40, 40, 1)))
            x += step
            matice[int(y / 50), int(x / 50) - 1] = predikce
        y += step
    return matice


def displayNumbers(img, numbers, solved_num, color=(0, 255, 0)):
    """
    :param img: transfomed image with sudoku grid
    :param numbers: matrix with predicted numbers
    :param solved_num: solved matrix
    :param color: color of numbers
    :return: image with solved sudoku
    """
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(9):
        for j in range(9):
            if numbers[j, i] == 0:
                cv2.putText(img, str(solved_num[j, i]),
                            (i * W + int(W / 2) - int((W / 4)), int((j + 0.7) * H)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color,
                            1, cv2.LINE_AA)
    return img


def get_InvPerspective(img, masked_num, location, height=450, width=450):
    """
    :param img: original image
    :param masked_num: transformed image with solved sudoku
    :param location: corners coordinations of original image
    :param height: height = 450
    :param width: width = 450
    :return: original image with solve sudoku
    """

    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[1], location[2], location[3]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1],
                                                      img.shape[0]))
    return result
