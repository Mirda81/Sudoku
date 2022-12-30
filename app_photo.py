import numpy as np
import cv2

from keras.models import load_model
from image_processing import perspective_transform, get_corners, draw_corners, text_on_top, seraching_rectange

from process import check_contour, predict, inv_transformation
from functions import load_sudoku_images

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
for i in range(0,obrazky.shape[0]):

    priklad = obrazky[i, :, :, :]
    model = load_model('model3.h5')
    contour_exist, prep_img, frame, contour, contour_line, thresh = check_contour(priklad)
    corners = get_corners(contour)
    result = perspective_transform(frame, (450, 450), corners)
    img_nums, centered_numbers, predicted_matrix, solved_matrix, time = predict(result, model)

    mask = np.zeros_like(result)
    priklad, img_solved = inv_transformation(mask, priklad, predicted_matrix, solved_matrix, corners)

    cv2.imwrite(f'solved_images/pic{i}.jpeg', priklad)

cv2.imshow("Final result", priklad)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


