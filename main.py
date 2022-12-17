import numpy as np
from keras.models import load_model


from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers,displayNumbers, get_InvPerspective, center_numbers, get_corners

from My_solver import solve_sudoku
import cv2
import time as t

prev = 0

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)
model = load_model('model2.h5')
# change brightness to 150
cap.set(10, 150)
seen = False
seen_corners = 0
not_seen_corners = 0
time_out_corners =3
while True:
    time_elapsed = t.time() - prev
    if time_elapsed > 1. / frame_rate:
        success, img = cap.read()
        img_result = img.copy()
        prev = t.time()

        prep_img = Preprocess(img_result)
        frame, contour= extract_frame(prep_img)

        if len(contour) == 4:

            corners = get_corners(contour)
            if seen_corners == 0:
                seen_corners = t.time()
            time_on_corners = t.time() - seen_corners
            cv2.putText(img=img_result, text=str(2-time_on_corners), org=(0, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7,
                        color=(0, 255, 0), thickness=1)
            print(f"time_on_corners: {time_on_corners}")
            for corner in corners:
                x, y = corner
                cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)


            if time_on_corners > 2:
                result = Perspective_transform(frame, (450, 450), corners)
                img_nums, stats, centroids = extract_numbers(result)
                matice = np.zeros((9, 9), dtype='uint8')
                not_seen_corners = 0

                if seen == False:
                    matice_predicted = predict_numbers(img_nums, matice, model)
                    matice_solved = matice_predicted.copy()
                    matice_predesla = np.sum(matice_predicted)
                    matice_solved = solve_sudoku(matice_solved)
                    seen = True

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
