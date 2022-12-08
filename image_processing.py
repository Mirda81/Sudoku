import cv2
import numpy as np
import keras
import pickle


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

def extract_numbers(img):
    result2 =cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,21,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.morphologyEx(result2, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel = np.ones((1,6),np.uint8)
    edges = cv2.dilate(gray,kernel,iterations = 1)
    kernel2 = np.ones((6,1),np.uint8)
    edges = cv2.dilate(edges,kernel2,iterations = 1)

    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(edges, [c], -1, (0,255,0), -1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    mrizka = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    mrizka = cv2.morphologyEx(mrizka, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    filtered_number = result2 - mrizka
    filtered_number[filtered_number == 1] = 0
    return filtered_number

def proccesCell(cell):
  kernel = np.ones((3,3),np.uint8)
  closing = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
  cropped_img = closing[5:closing.shape[0]-5, 5:closing.shape[0]-5]
  cc = center_crop(cropped_img,(40,40))
  ret,thresh = cv2.threshold(cc, 125, 255,
                       cv2.THRESH_BINARY_INV)
  return thresh

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2)
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def predict_numbers(numbers, matice, model):
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
                # vysek = cv2.resize(vysek,(28,28))
                vysek = vysek / 255
                predikce = np.argmax(model.predict(vysek.reshape(1, 40, 40, 1)))
            x += step
            matice[int(y/50),int(x/50)-1] = predikce
        y += step
    return matice

def displayNumbers(img, numbers, solved_num, color=(0, 255, 0)):
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range (9):
        for j in range (9):
            if numbers[j,i] ==0:
                cv2.putText(img, str(solved_num[j,i]),
                (i*W+int(W/2)-int((W/4)),int((j+0.7)*H)),
                cv2.FONT_HERSHEY_COMPLEX, 1, color,
                1, cv2.LINE_AA)
    return img

def get_InvPerspective(img, masked_num, location, height = 450, width = 450):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1],
    img.shape[0]))
    return result