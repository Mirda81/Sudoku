import os, random
from functions import load_sudoku_images
from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers
import cv2
import numpy as np
import time as t

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
priklad = obrazky[0,:,:,:]
prev = 0



prep_img = Preprocess(priklad)
frame, contour = extract_frame(prep_img)
result = Perspective_transform(frame,(450,450), contour)
numbers = extract_numbers(result)
matice = np.zeros((9,9),np.uint8)
matice_cisla = predict_numbers(numbers, matice)
print(matice_cisla)
cv2.imshow('window', numbers)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


