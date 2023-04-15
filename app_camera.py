import cvzone
import time as t
from keras.models import load_model
from image_processing import perspective_transform, get_corners, draw_corners, text_on_top, seraching_rectange
from functions import camera_set
from process import check_contour, predict, inv_transformation
from text_conditions import get_vars, dots

from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

output_size = (800, 600)
cap = cv2.VideoCapture(0)

def gen_frames():  # function that generates frames to be displayed

    output_size = (800, 600)
    cap = camera_set(output_size[0], output_size[1])
    # load model for predict numbers
    model = load_model('model3.h5')
    # load frame image
    bkg = cv2.imread('pngegg.png', cv2.IMREAD_UNCHANGED)
    bkg = cv2.resize(bkg, output_size)

    # pre set variables
    prev = 0
    seen = False
    steps_mode = False
    bad_read = False
    solved = False
    seen_corners = 0
    not_seen_corners = t.time() - 1
    wait = 0.4
    process_step = 0
    rectangle_counter = 0
    time_on_corners = 0
    dots_str = ""
    time = ""

    while True:
        time_elapsed = t.time() - prev

        start = t.time()
        prev = t.time()
        success, img = cap.read()
        img = cv2.resize(img, output_size)
        img_result = img.copy()

        # apply threshold to image
        contour_exist, prep_img, frame, contour, contour_line, thresh = check_contour(img_result)
        # if countour exist, wait 2 seconds - time to focus grid properly
        if contour_exist:
            corners = get_corners(contour)
            # draw corners on image
            not_seen_corners = 0
            out_corners_check = False
            if not solved:
                if not bad_read:
                    color = (0, 0, 255) if int((10 * time_on_corners)) % 3 == 0 else (0, 255, 0)
                cv2.drawContours(img_result, [contour], -1, color, 2)
            else:
                draw_corners(img_result, corners)

            # start timer when we see a grid
            if seen_corners == 0:
                seen_corners = t.time()
            time_on_corners = t.time() - seen_corners

            # 0,4 sec limit to focus properly, start main cycle(transfomation, recognition, solving, wrap)
            if time_on_corners > wait:
                wait = 0.4
                dots_str = ''
                # make a perspective transformation
                transformed_size = (450, 450)
                result = perspective_transform(frame, transformed_size, corners)
                # if grid was not seen already predict numbers and solve
                if not seen:
                    # if contour seen for the first time solve puzzle, getting all steps images to be able put them to speps mode
                    img_nums, centered_numbers, predicted_matrix, solved_matrix, time = predict(result, model)
                    # check if sudoko was solved succesfully
                    if np.any(solved_matrix == 0):
                        # seen_corners = 0
                        bad_read = True
                        solved = False
                    else:
                        bad_read = False
                        seen = True
                        solved = True
                        wait = 0.03

                # make an inverse transormation
                if not bad_read:
                    mask = np.zeros_like(result)
                    img_result, img_solved = inv_transformation(mask, img_result, predicted_matrix, solved_matrix,
                                                                corners)

        # if we dont see corners, keep 0.2) seconds old solution then ready to new solution (to avoid flashing when we move with puzzle
        else:
            if not_seen_corners == 0:
                not_seen_corners = t.time()
            time_out_corners = t.time() - not_seen_corners
            out_corners_check = time_out_corners > 0.2
            # if we pass waiting time out corners reset variables
            if out_corners_check:
                dots_str = dots(time_out_corners)
                seen = False
                seen_corners = 0
                bad_read = False
                solved = False
                wait = 0.4
                img_result, corner_rect = seraching_rectange(img_result, rectangle_counter)
                # helping variable for searching rectangle
                if corner_rect > 200:
                    rectangle_counter = -1
                rectangle_counter += 1

        # text writing
        fps = int(1 / time_elapsed)
        text, pos, color1 = get_vars(out_corners_check, solved, bad_read, time_on_corners, seen, time)
        text_on_top(img_result, text + dots_str, color1, pos, fps)
        # steps mode, enter step mode by press "m", then navigate with press 1,0 esc  to quit

        img_result = cvzone.overlayPNG(img_result, bkg, [0, 0])


        # convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img_result)
        img_result = buffer.tobytes()

        # use yield to send the frame to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_result + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
