import numpy as np
from keras.models import load_model

from functions import load_sudoku_images
from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers,displayNumbers, get_InvPerspective, center_numbers, get_corners
from solver import solve
from My_solver import solve_sudoku
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
model = load_model('model.h5')
# change brightness to 150
cap.set(10, 150)
matice_predesla = 0
predesle_corners = 0
while True:

    time_elapsed = t.time() - prev


    if time_elapsed > 1. / frame_rate:
        success, img = cap.read()
        img_result = img.copy()
        prev = t.time()

        prep_img = Preprocess(img_result)
        # cv2.imshow('res2', prep_img)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        frame, contour, res2 = extract_frame(prep_img)

        corners = get_corners(contour)
        corners_check = int(round(np.sum(corners),-1)) == predesle_corners

        for corner in corners:
            x, y = corner
            cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)

        if not corners_check:
            predesle_corners = int(round(np.sum(corners),-1))

            result = Perspective_transform(frame, (450, 450), corners)
            img_nums, stats, centroids = extract_numbers(result)
            matice = np.zeros((9, 9), dtype='uint8')
            matice_predicted = predict_numbers(img_nums, matice, model)


            matice_solved = matice_predicted.copy()
            matice_predesla = np.sum(matice_predicted)
            matice_solved = solve_sudoku(matice_solved)

            mask = np.zeros_like(result)
            img_solved = displayNumbers(mask, matice_predicted, matice_solved)

            inv = get_InvPerspective(img_result, img_solved, corners)


        combined = cv2.addWeighted(img_result, 0.7, inv, 1, -1)

        cv2.imshow('window', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
