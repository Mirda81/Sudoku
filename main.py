import os, random
from functions import load_sudoku_images
from image_processing import Preprocess, extract_frame, Perspective_transform, frame_kont, extract_numbers
import cv2
import time as t

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
priklad = obrazky[0,:,:,:]
prev = 0

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)
success, img = cap.read()
while True:
    time_elapsed = t.time() - prev

    success, img = cap.read()
    img_result = img.copy()
    prep_img = Preprocess(img_result)
    frame, contour = extract_frame(prep_img)
    result = Perspective_transform(frame,(450,450), contour)
    kontura = frame_kont(prep_img)
    cv2.drawContours(img, kontura, -1, (0, 255, 0), 2)
    cv2.imshow('window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
