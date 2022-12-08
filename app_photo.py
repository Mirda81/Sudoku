import os, random
from functions import load_sudoku_images
from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers, displayNumbers, get_InvPerspective
from solver import solve, Solved
import cv2
import numpy as np
from keras.models import load_model
import pickle
import time as t

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
priklad = obrazky[2,:,:,:]
prev = 0
#model = pickle.load(open('model.pkl', 'rb'))

model = load_model('model.h5')
prep_img = Preprocess(priklad)
frame, contour = extract_frame(prep_img)
result = Perspective_transform(frame,(450,450), contour)
numbers = extract_numbers(result)
matice = np.zeros((9,9),np.uint8)
matice_cisla = predict_numbers(numbers, matice, model)
vyresena_cisla = matice_cisla.copy()
mask = np.zeros_like(result)

print(matice_cisla)
if solve(vyresena_cisla):
    print(vyresena_cisla)
else:
    print("Solution don't exist. Model misread digits.")
reseni = displayNumbers(mask,matice_cisla,vyresena_cisla)
inv = get_InvPerspective(priklad, reseni, contour)
combined = cv2.addWeighted(priklad, 0.7, inv, 1, 0)
cv2.imshow("Final result", combined)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


