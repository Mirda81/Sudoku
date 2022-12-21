import numpy as np
import cvzone
from keras.models import load_model
from image_processing import preprocess, extract_frame, perspective_transform, extract_numbers, predict_numbers, \
    displayNumbers, get_inv_perspective, center_numbers, get_corners, draw_corners, write_text
from functions import camera_set

from My_solver import solve_sudoku
import cv2
import time as t

# camera setting
frameWidth = 800
frameHeight = 600
frame_rate = 30
cap = camera_set(frameWidth, frameHeight)

result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (frameWidth, frameHeight))
# load frame image
bkg = cv2.imread('pngegg.png', cv2.IMREAD_UNCHANGED)
bkg = cv2.resize(bkg, (800, 600))

prev = 0
model = load_model('model2.h5')

seen = False
limit_on_cornes = 2
seen_corners = 0
not_seen_corners = t.time() - 2
time_out_corners = 1
time_for_recognition = 0
bad_read = False
text1 = "Ready to new recognition"
text2 = ""
color1 = (0, 255, 125)
color2 = (0, 255, 125)
success, img = cap.read()
img_result = img.copy()
solved = False
contour_prev = []
pocitadlo = 0
nasobek = 1
steps_mode = False
promenna = 0
rectangle_counter = 0
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    time_elapsed = t.time() - prev
    if time_elapsed > 1. / frame_rate:
        success, img = cap.read()
        img_result = img.copy()
        prev = t.time()
        # apply threshold to image
        prep_img = preprocess(img_result)
        # find biggest 4 side contour and extract sudoku grid

        frame, contour, contour_line, thresh = extract_frame(prep_img)
        contour_exist = len(contour) == 4
        # if countour exist, wait 2 seconds - time to focus grid properly

        if contour_exist:
            corners = get_corners(contour)
            # draw corners on image

            if not solved:
                cv2.drawContours(img_result, [contour], -1, (0, 255, 0), 2)
            else:
                draw_corners(img_result, corners)

            # start timer when we see a grid
            if seen_corners == 0:
                seen_corners = t.time()
            time_on_corners = t.time() - seen_corners
            # when model misread digit before, give 3 sec to adjust camera before new recognition cycle
            if time_on_corners < limit_on_cornes:
                if bad_read:
                    text1 = 'model misread digits'
                    text2 = "new digit recocgnition starts in " + str(round(3 - time_on_corners, 2))
                else:
                    text1 = "Sudoku frame detected"
                    text2 = "digit recocgnition starts in " + str(round(2 - time_on_corners, 2))
                    color = (0, 255, 0)

            print(f"time_on_corners: {time_on_corners}")

            # if we reach 2 sec limit to focus, start main cycle(transfomation, recognition, solving, wrap)
            if time_on_corners > limit_on_cornes:
                nasobek = 1
                not_seen_corners = 0
                # make a perspective transformation
                result = perspective_transform(frame, (450, 450), corners)
                # if grid was not seen already predict numbers and solve
                if not seen:
                    img_nums, stats, centroids = extract_numbers(result)
                    centered_numbers = center_numbers(img_nums, stats, centroids)
                    empty_matrix = np.zeros((9, 9), dtype='uint8')

                    start_predicition = t.time()
                    predicted_matrix = predict_numbers(centered_numbers, empty_matrix, model)
                    end_prediction = t.time()

                    solved_matrix = predicted_matrix.copy()
                    start = t.time()
                    solved_matrix = solve_sudoku(solved_matrix)
                    end = t.time()
                    # check if sudoko was solved succesfuly
                    if np.any(solved_matrix == 0):
                        seen_corners = 0
                        bad_read = True
                        color = (0, 0, 255)
                        limit_on_cornes = 3
                        solved = False
                    else:
                        text1 = 'Solved in ' + str(round(end - start, 3)) + ' s'
                        text2 = ''
                        # text2 = "Digits recognized in " + str(round(end_prediction - start_predicition, 3)) + ' s'
                        color = (0, 255, 0)
                        pos1 = (265 ,30)
                        bad_read = False
                        seen = True
                        limit_on_cornes = 2
                        solved = True

                # make an inverse transormation
                mask = np.zeros_like(result)
                img_solved = displayNumbers(mask, predicted_matrix, solved_matrix)
                inv = get_inv_perspective(img_result, img_solved, corners)
                img_result = cv2.addWeighted(img_result, 1, inv, 1, -1)

        # if we dont see corners, keep 2 seconds old solution then ready to new solution
        else:
            if not_seen_corners == 0:
                not_seen_corners = t.time()
            time_out_corners = t.time() - not_seen_corners
            print(f"time_out_corners: {time_out_corners}")
            if time_out_corners > 2:
                multiplier = int(time_out_corners // 1)
                if multiplier > (5 * nasobek):
                    nasobek += 1
                tecky = 5 + multiplier - (5 * nasobek)
                text2 = "Searching for grid" + '.' * tecky
                text1 = "ready to new recognition"
                color2 = (125, 0, 255)
                seen = False
                seen_corners = 0
                solved = False
                corner_1 = (75+(3*rectangle_counter), 75+(3*rectangle_counter))
                corner_2 =  (725-(3*rectangle_counter), 525-(3*rectangle_counter))
                cv2.rectangle(img_result,  corner_1, corner_2, (0, 0, 255), 2)
                if corner_1[0] > 200:
                    rectangle_counter = -1
                rectangle_counter+=1
        # text writing

        write_text(img_result,text1,color1,(250, 30),text2,color2,(275, 60))        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #      break

        if solved and steps_mode:

            img2 = img.copy()
            detected_frame = cv2.drawContours(img2, [contour], -1, (0, 255, 0), 2).copy() if contour_exist else img2
            process = [img, prep_img, thresh, detected_frame, contour_line, frame, result, img_nums, centered_numbers,
                       img_solved, img_result]

            key = cv2.waitKey(1)
            if int(key) == 48:
                promenna = max(0,promenna-1)
            if int(key) == 49:
                promenna = min(len(process)-1 , promenna + 1)
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
            key =cv2.waitKey(1)
            if  int(key) == 109:
                steps_mode = True
            if int(key) == 113:
                break
cap.release()
cv2.destroyAllWindows()
