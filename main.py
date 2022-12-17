import numpy as np
from keras.models import load_model


from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers,displayNumbers, get_InvPerspective, center_numbers, get_corners

from My_solver import solve_sudoku
import cv2
import time as t

prev = 0

frameWidth = 1280
frameHeight = 1024

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
time_out_corners =3
bad_read = False
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    time_elapsed = t.time() - prev
    if time_elapsed > 1. / frame_rate:
        success, img = cap.read()
        img_result = img.copy()
        prev = t.time()

        prep_img = Preprocess(img_result)
        frame, contour= extract_frame(prep_img)

        if len(contour) == 4:
            not_seen_corners = 0
            corners = get_corners(contour)
            if seen_corners == 0:
                seen_corners = t.time()
            time_on_corners = t.time() - seen_corners
            if (time_on_corners < limit_on_cornes):
                if bad_read:
                    text1 = 'model misread digits'
                    text2 = "new digit recocgnition starts in " + str(round(3-time_on_corners,2))
                else:
                    text1 = "Sudoku frame detected"
                    text2 = "digit recocgnition starts in " + str(round(2-time_on_corners,2))
                    color = (0, 255, 0)

            cv2.putText(img=img_result, text=text1, org=(0, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7,
                        color=color, thickness=1)
            cv2.putText(img=img_result, text= text2, org=(0, 125), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6,
                        color=color, thickness=1)
            print(f"time_on_corners: {time_on_corners}")
            for corner in corners:
                x, y = corner
                cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)


            if time_on_corners > limit_on_cornes:
                result = Perspective_transform(frame, (450, 450), corners)
                img_nums, stats, centroids = extract_numbers(result)

                if seen == False:
                    centered_numbers = center_numbers(img_nums, stats, centroids)
                    matice = np.zeros((9, 9), dtype='uint8')
                    not_seen_corners = 0
                    matice_predicted = predict_numbers(centered_numbers, matice, model)
                    matice_solved = matice_predicted.copy()
                    start = t.time()
                    matice_solved = solve_sudoku(matice_solved)
                    end = t.time()
                    if np.any(matice_solved ==0):
                        seen_corners = 0
                        bad_read = True
                        color = (0, 0, 255)
                        limit_on_cornes = 3
                    else:
                        text1 = 'Solved in ' + str(round(end - start,3)) + ' s'
                        text2 = ""
                        color = (0, 255, 0)
                        bad_read = False
                        seen = True
                        limit_on_cornes = 2

                mask = np.zeros_like(result)
                img_solved = displayNumbers(mask, matice_predicted, matice_solved)
                inv = get_InvPerspective(img_result, img_solved, corners)


                combined = cv2.addWeighted(img_result, 0.7, inv, 1, -1)

                cv2.imshow('window', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.imshow('window', img_result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            if not_seen_corners == 0:
                not_seen_corners = t.time()
            time_out_corners = t.time() - not_seen_corners
            print(f"time_out_corners: {time_out_corners}")
            if time_out_corners > 2:
                seen = False
                seen_corners = 0
            cv2.imshow('window', img_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()
