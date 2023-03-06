import cvzone
import cv2
from image_processing import preprocess, extract_frame,extract_numbers,center_numbers,predict_numbers,displayNumbers,get_inv_perspective
from Solver_final import solve_wrapper

def check_contour(img,thresh,threshold_method,block_size,c):
    prep_img = preprocess(img,(3,3))
    frame, contour, contour_line, thresh = extract_frame(prep_img, thresh,threshold_method,block_size,c)
    contour_exist = len(contour) == 4

    return contour_exist,prep_img, frame, contour, contour_line, thresh

def predict(img,model):
    img_nums, stats, centroids, nums = extract_numbers(img,'ADAPTIVE_THRESH_GAUSSIAN_C')
    centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
    predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)
    solved_matrix, time = solve_wrapper(predicted_matrix.copy())

    return img_nums, centered_numbers, predicted_matrix, solved_matrix, time

def inv_transformation(mask,img,predicted_matrix,solved_matrix,corners):
    img_solved = displayNumbers(mask, predicted_matrix, solved_matrix)
    inv = get_inv_perspective(img, img_solved, corners)
    img = cv2.addWeighted(img,1, inv,1, 0,-1)
    return img,img_solved

