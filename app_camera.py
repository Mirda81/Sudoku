import numpy as np
import cvzone
from keras.models import load_model
from image_processing import preprocess, extract_frame, perspective_transform, extract_numbers, predict_numbers, \
    displayNumbers, get_inv_perspective, center_numbers, get_corners, draw_corners, text_on_top, bottom_text
from functions import camera_set
from Sudoku_solver import solve
from test import solve_wrapper

from My_solver import solve_sudoku
import cv2
import time as t

# camera setting
output_size = (800,600)
frame_rate = 40
cap = camera_set(output_size[0], output_size[1])

# load frame image
bkg = cv2.imread('pngegg.png', cv2.IMREAD_UNCHANGED)
bkg = cv2.resize(bkg, output_size)

prev = 0
model = load_model('model3.h5')

seen = False
limit_on_cornes = 1
seen_corners = 0
not_seen_corners = t.time() - 1
time_out_corners = 1
time_for_recognition = 0
bad_read = False
color1 = (0, 255, 0)
color2 = (0, 255, 0)
success, img = cap.read()
img_result = img.copy()
solved = False
contour_prev = []
pocitadlo = 0
nasobek = 1
steps_mode = False
promenna = 0
rectangle_counter = 0
wait = 0.8
time_on_corners = 0

pos1 = (320, 30)
pos2= (275, 60)
# cv2.namedWindow("sudoku solver", cv2.WND_PROP_FULLSCREEN)
# # cv2.setWindowProperty("sudoku solver",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, output_size)

while (cap.isOpened()):
    time_elapsed = t.time() - prev

    start = t.time()
    prev = t.time()
    success, img = cap.read()
    img_result = img.copy()
    # apply threshold to image
    prep_img = preprocess(img_result)
    # find biggest 4 side contour and extract sudoku grid

    frame, contour, contour_line, thresh = extract_frame(prep_img)
    contour_exist = len(contour) == 4
    # if countour exist, wait 2 seconds - time to focus grid properly

    if contour_exist:
        corners = get_corners(contour)
        # draw corners on image
        not_seen_corners = 0

        if not solved:
            if bad_read:
                color1 = (0, 0, 255)
                color2 = (0, 140, 255)
                if time_on_corners > 2:
                    text1 = 'model misread digits'
                    text2 = ""
                    color1=(0,0,255)
                wait = 0.5
            else:
                text1 = "sudoku grid detected"
                text2 = ""
                color = (0, 0, 255) if int((10 * time_on_corners)) % 3 == 0 else (0, 255, 0)
                color1 = (0,255,0)
            cv2.drawContours(img_result, [contour], -1, color, 2)
        else:
            draw_corners(img_result, corners)

        # start timer when we see a grid
        if seen_corners == 0:
            seen_corners = t.time()
        time_on_corners = t.time() - seen_corners

        print(f"time_on_corners: {time_on_corners}")

        # if we reach 2 sec limit to focus, start main cycle(transfomation, recognition, solving, wrap)
        if time_on_corners > wait:
            wait = 0.8
            nasobek = 1

            # make a perspective transformation
            result = perspective_transform(frame, (450, 450), corners)
            # if grid was not seen already predict numbers and solve
            if not seen:
                img_nums, stats, centroids = extract_numbers(result)
                centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
                empty_matrix = np.zeros((9, 9), dtype='uint8')
                # cv2.imshow('numbers', centered_numbers)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                start_predicition = t.time()
                predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)
                end_prediction = t.time()

                solved_matrix = predicted_matrix.copy()
                solved_matrix, time = solve_wrapper(solved_matrix)
                print(solved_matrix)

                # check if sudoko was solved succesfuly
                if np.any(solved_matrix == 0):
                    # seen_corners = 0
                    bad_read = True
                    color1 = (0, 144, 255)
                    solved = False
                else:
                    text1 = time
                    # text2 = "Digits recognized in " + str(round(end_prediction - start_predicition, 3)) + ' s'
                    color1 = (0, 255, 0)
                    color2 = (0, 255, 0)
                    pos1 = (285, 30)
                    pos2 = (225,60)
                    bad_read = False
                    seen = True
                    solved = True
                    wait = 0.03

            # make an inverse transormation
            if not bad_read:
                mask = np.zeros_like(result)
                img_solved = displayNumbers(mask, predicted_matrix, solved_matrix)
                inv = get_inv_perspective(img_result, img_solved, corners)
                img_result = cv2.addWeighted(img_result, 1, inv, 1, -1)

    # if we dont see corners, keep 2 seconds old solution then ready to new solution
    else:
        if not_seen_corners == 0:
            not_seen_corners = t.time()
        time_out_corners = t.time() - not_seen_corners

        if time_out_corners > 0.4:
            multiplier = int(time_out_corners // 1)
            if multiplier > (5 * nasobek):
                nasobek += 1
            tecky = 5 + multiplier - (5 * nasobek)
            text1 = "Searching for grid" + '.' * tecky
            # text1 = "Ready"
            pos1=(315,25)
            color1 = (255, 255, 255)
            color2 = (125, 255, 125)
            seen = False
            seen_corners = 0
            solved = False
            wait = 0.8
            bad_read = False
            corner_1 = (75 + (3 * rectangle_counter), 75 + (3 * rectangle_counter))
            corner_2 = (725 - (3 * rectangle_counter), 525 - (3 * rectangle_counter))
            cv2.rectangle(img_result, corner_1, corner_2, (0, 0, 255), 2)
            if corner_1[0] > 200:
                rectangle_counter = -1
            rectangle_counter += 1
    # text writing

    text_on_top(img_result, text1, color1, pos1, text2, color2, pos2)
    cv2.putText(img=img_result, text=f'fps: {int(1/time_elapsed)}', org=(35, 60),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                color=color1, thickness=1)
    #      break
    # out.write(img_result)
    if solved and steps_mode:

        img2 = img.copy()
        detected_frame = cv2.drawContours(img2, [contour], -1, (0, 255, 0), 2).copy() if contour_exist else img2
        process = [img, prep_img, thresh, detected_frame, contour_line, frame, result, img_nums, centered_numbers,
                   img_solved, img_result]

        key = cv2.waitKey(1)
        if int(key) == 48:
            promenna = max(0, promenna - 1)
        if int(key) == 49:
            promenna = min(len(process) - 1, promenna + 1)
        if key == 27:
            steps_mode = False  # escape
        if int(key) == 113:
            break
        print(promenna)
        obrazek = process[int(promenna)]

        try:

            cv2.imshow('sudoku solver', obrazek)
        except:
            mask = np.zeros_like(img_result)
            cv2.imshow('sudoku solver', mask)

    else:
        img_result = cvzone.overlayPNG(img_result, bkg, [0, 0])
        cv2.imshow('sudoku solver', img_result)
        key = cv2.waitKey(1)
        if int(key) == 109:
            steps_mode = True
        if int(key) == 113:
            break

    print(t.time() - start)

cap.release()
cv2.destroyAllWindows()
