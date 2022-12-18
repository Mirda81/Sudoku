import numpy as np
from keras.models import load_model

from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers, \
    displayNumbers, get_InvPerspective, center_numbers, get_corners

from My_solver import solve_sudoku
import cv2
import time as t

bkg = cv2.imread('pic.png')
bkg = cv2.resize(bkg, (640, 480))
prev = 0

frameWidth = 1920
frameHeight = 1080

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)
model = load_model('model2.h5')
# change brightness to 150
cap.set(10, 150)
seen = False
limit_on_cornes = 2
seen_corners = 0
not_seen_corners = 0
time_out_corners = 3
time_for_recognition = 0
bad_read = False
text1 = "ready to new recognition"
text2 = ""
color = (255, 0, 0)
success, img = cap.read()
img_result = img.copy()
promenna = 7
solved = False
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    time_elapsed = t.time() - prev
    if time_elapsed > 1. / frame_rate:
        success, img = cap.read()
        img_result = img.copy()
        print(img_result.shape, bkg.shape)
        prev = t.time()
        # apply threshold to image
        prep_img = Preprocess(img_result)
        # find biggest 4 side contour and extract sudoku grid
        frame, contour, contour_line = extract_frame(prep_img)
        # if countour exist, wait 2 seconds - time to focus grid properly
        if len(contour) == 4:
            corners = get_corners(contour)
            # draw corners on image
            for corner in corners:
                x, y = corner
                cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)
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
                not_seen_corners = 0
                # make a perspective transformation
                result = Perspective_transform(frame, (450, 450), corners)
                # if grid was not seen already predict numbers and solve
                if not seen:
                    img_nums, stats, centroids = extract_numbers(result)
                    centered_numbers = center_numbers(img_nums, stats, centroids)
                    empty_matrix = np.zeros((9, 9), dtype='uint8')
                    not_seen_corners = 0
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
                        text2 = "Digits recognized in " + str(round(end_prediction - start_predicition, 3)) + ' s'
                        color = (0, 255, 0)
                        bad_read = False
                        seen = True
                        limit_on_cornes = 2
                        solved = True

                # make an inverse transormation
                mask = np.zeros_like(result)
                img_solved = displayNumbers(mask, predicted_matrix, solved_matrix)
                inv = get_InvPerspective(img_result, img_solved, corners)
                img_result = cv2.addWeighted(img_result, 1, inv, 1, -1)

        # if we dont see corners, keep 2 seconds old solution then ready to new solution
        else:
            if not_seen_corners == 0:
                not_seen_corners = t.time()
            time_out_corners = t.time() - not_seen_corners
            print(f"time_out_corners: {time_out_corners}")
            if time_out_corners > 2:
                text1 = "ready to new recognition"
                color = (255, 0, 0)
                seen = False
                seen_corners = 0
                solved = False

        # text writing
        cv2.rectangle(img_result, (0, 0), (1000, 60), (0, 0, 0), -1)
        cv2.putText(img=img_result, text=text1, org=(0, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7,
                    color=color, thickness=1)
        cv2.putText(img=img_result, text=text2, org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6,
                    color=color, thickness=1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #      break
        if solved:
            process = [img, contour_line, frame, result, img_nums, centered_numbers, img_solved, img_result]
            key = cv2.waitKey(1)
            if int(key) in range(49, 57):
                promenna = int(key) - 49
            if key == 27:
                break  # escape

            obrazek = process[int(promenna)]
            cv2.imshow('window', obrazek)
        else:
            cv2.imshow('window', img_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
