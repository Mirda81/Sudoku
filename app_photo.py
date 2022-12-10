import cv2
import numpy as np
from keras.models import load_model

from functions import load_sudoku_images
from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers,displayNumbers, get_InvPerspective, center_numbers, get_corners
from solver import solve

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
priklad = obrazky[4,:,:,:]
prev = 0
#model = pickle.load(open('model.pkl', 'rb'))
model = load_model('model.h5')
prep_img = Preprocess(priklad)
frame, contour, res2 = extract_frame(prep_img)
corners = get_corners(contour)
result = Perspective_transform(frame,(450,450), corners)
img, stats, centroids = extract_numbers(result)
centered_numbers = center_numbers(img, stats, centroids)

matice = np.zeros((9,9), dtype='uint8')
matice_predicted = predict_numbers(img,matice,model)
matice_solved = matice_predicted.copy()
print(matice_predicted)

# cv2.imshow("transormed", priklad)
# cv2.imshow("priklad", result)
# cv2.imshow("Final result", centered_numbers)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

if solve(matice_solved):
    print(matice_solved)
else:
    print("Solution don't exist. Model misread digits.")
mask = np.zeros_like(result)
img_solved = displayNumbers(mask,matice_predicted,matice_solved)



print(matice_predicted)
# if solve(vyresena_cisla):
#     print(vyresena_cisla)
# else:
#     print("Solution don't exist. Model misread digits.")
# reseni = displayNumbers(mask,matice_cisla,vyresena_cisla)
inv = get_InvPerspective(priklad, img_solved, corners)
combined = cv2.addWeighted(priklad, 0.7, inv, 1, -1)
cv2.imshow("Final", centered_numbers)
cv2.imshow("Final result", combined)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


