import numpy as np
import cvzone
import cv2
import time as t
from keras.models import load_model
from image_processing import perspective_transform,predict_numbers,get_corners, draw_corners, text_on_top, seraching_rectange,preprocess,extract_frame, extract_numbers,center_numbers
from functions import camera_set
from process import check_contour, predict, inv_transformation
from text_conditions import get_vars, dots
import streamlit as st
from Solver_final import solve_wrapper

def video():
    # camera setting
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
    dots_str= ""
    time = ""

    while (cap.isOpened()):
        time_elapsed = t.time() - prev
        start = t.time()
        prev = t.time()
        success, img = cap.read()
        img = cv2.resize(img, output_size)
        img_result = img.copy()
        frame_img = img_result
        prep_img = preprocess(img_result,(blur,blur))
        frame, contour, contour_line,thresh = extract_frame(prep_img, threshold, st.session_state['threshold_method'], block_size, c)
        st.session_state.frame4 = empty_frame
        st.session_state.frame6 = empty_frame
        st.session_state.frame7 = empty_frame
        st.session_state.frame8 = np.zeros((450, 450, 3), dtype=np.uint8)
        # apply threshold to image
        contour_exist = len(contour) == 4

        # if countour exist, wait 2 seconds - time to focus grid properly
        if contour_exist:
            corners = get_corners(contour)
            # draw corners on image
            not_seen_corners = 0
            out_corners_check = False
            if not solved:
                if not bad_read:
                    color = (0, 255, 0)
                    frame_img = cv2.drawContours(img_result, [contour], -1, color, 2)
            else:
                draw_corners(img_result, corners)

            # start timer when we see a grid
            if seen_corners == 0:
                seen_corners = t.time()
            time_on_corners = t.time() - seen_corners

            # 0,4 sec limit to focus properly, start main cycle(transfomation, recognition, solving, wrap)
            if time_on_corners > wait:
                wait = 0.4
                dots_str= ''
                # make a perspective transformation
                transformed_size = (450, 450)
                result = perspective_transform(frame, transformed_size, corners)
                st.session_state.frame4 = frame
                # if grid was not seen already predict numbers and solve
                if not seen:
                    # if contour seen for the first time solve puzzle, getting all steps images to be able put them to speps mode
                    img_nums, stats, centroids, thresh_nums = extract_numbers(result,st.session_state['threshold_method'])
                    st.session_state.frame6 = result
                    st.session_state.frame7 = thresh_nums

                    centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
                    st.session_state.frame8 = img_nums
                    # st.session_state.frame5 = centered_numbers
                    # check if sudoko was solved succesfully
                    # predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)
                    # solved_matrix, time = solve_wrapper(predicted_matrix.copy())
                    # if np.any(solved_matrix == 0):
                    #     # seen_corners = 0
                    #     bad_read = True
                    #     solved = False
                    # else:
                    #     bad_read = False
                    #     seen = True
                    #     solved = True
                    #     wait = 0.03

                # make an inverse transormation
                # if not bad_read:
                #     mask = np.zeros_like(result)
                #     img_result, img_solved = inv_transformation(mask, img_result, predicted_matrix, solved_matrix, corners)

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
                # helping variable for searching rectangle
        # text writing
        fps = int(1 / time_elapsed)
        text, pos, color1 = get_vars(out_corners_check,solved,bad_read,time_on_corners,seen,time)
        text_on_top(img_result, text + dots_str, color1, pos, fps)
        # steps mode, enter step mode by press "m", then navigate with press 1,0 esc  to quit


        img_result = cvzone.overlayPNG(img_result, bkg, [0, 0])
        st.session_state.frame1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB )
        st.session_state.frame2 = thresh
        st.session_state.frame3 = prep_img
        st.session_state.frame5 = frame_img


        my_slot1.image(st.session_state.frame1, channels="RGB", use_column_width=True)
        my_slot2.image(st.session_state.frame2, use_column_width=True)
        my_slot5.image(st.session_state.frame5, use_column_width=True)
        my_slot3.image(st.session_state.frame3, use_column_width=True)
        my_slot4.image(st.session_state.frame4,channels="RGB", use_column_width=True)
        my_slot6.image(st.session_state.frame6, channels="RGB",width=500, use_column_width=True)
        my_slot7.image(st.session_state.frame7, channels="RGB", width=500, use_column_width=True)
        my_slot8.image(st.session_state.frame8, channels="RGB", width=500, use_column_width=True)
    cap.release()
    cv2.destroyAllWindows()

import streamlit as st
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: blue;'>Playground</h1>", unsafe_allow_html=True)
empty_frame = np.zeros((600, 800, 3), dtype=np.uint8)

# Nastavení tématu pomocí konfiguračního souboru

if 'video_started' not in st.session_state:
    st.session_state['video_started'] = False

if 'frame1' not in st.session_state:
    st.session_state['frame1'] = empty_frame

if 'frame2' not in st.session_state:
    st.session_state['frame2'] = empty_frame

if 'frame3' not in st.session_state:
    st.session_state['frame3'] = empty_frame

if 'frame4' not in st.session_state:
    st.session_state['frame4'] = empty_frame

if 'frame5' not in st.session_state:
    st.session_state['frame5'] = empty_frame

if 'frame6' not in st.session_state:
    st.session_state['frame6'] = empty_frame

if 'frame7' not in st.session_state:
    st.session_state['frame'+ str(7)] = empty_frame

if 'frame8' not in st.session_state:
    st.session_state['frame8'] = empty_frame

col1, col2,col3, col4,col5, col6 = st.columns([20,300,20,300,20,300])

with st.sidebar:
    st.write("Blurring")
    blur = st.slider('Blur', 1, 11, 3, step=2)
    st.write("Thresholding")
    threshold = st.slider('Threshold', 0, 255, 255)
    st.session_state['threshold_method'] = st.selectbox('Threshold method', ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C'])
    block_size = st.slider("Block Size", 3, 21, 9, step=2)
    c = st.slider("C", -10, 10, 5, key='c_slider')

    # Create a checkbox and get its value
    checkbox_value = st.checkbox('Center image to grid')

    # Save the checkbox value
    if checkbox_value:
        checkbox_state = 'checked'
    else:
        checkbox_state = 'unchecked'

with col2:
    st.subheader("original image")
    my_slot1 = st.empty()
    my_slot1.image(st.session_state.frame1, width=500, use_column_width=True)

    st.subheader("centered numbers")
    my_slot5 = st.empty()
    my_slot5.image(st.session_state.frame5, width=500, use_column_width=True)

    st.subheader("threshold numbers")
    my_slot6 = st.empty()
    my_slot6.image(st.session_state.frame6, width=500, use_column_width=True)
# Display the third image in the second column
with col4:
    st.subheader("blured and gryscale")
    my_slot3 = st.empty()
    my_slot3.image(st.session_state.frame3, width=500, use_column_width=True)
# Display the fourth image in the second column

    st.subheader("result image")
    my_slot4 = st.empty()
    my_slot4.image(st.session_state.frame4, width=500, use_column_width=True)

    st.subheader("result image")
    my_slot7 = st.empty()
    my_slot7.image(st.session_state.frame7, width=500, use_column_width=True)

with col6:
    st.subheader("threshold frame")
    my_slot2 = st.empty()
    my_slot2.image(st.session_state.frame2, width=500, use_column_width=True)

    st.subheader("nums image")
    my_slot8 = st.empty()
    my_slot8.image(st.session_state.frame8, width=500, use_column_width=True)




while True:
    video()
