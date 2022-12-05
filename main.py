import os, random
from functions import load_sudoku_images
from image_processing import Preprocess
import cv2

slozka_sudoku = r'test_imgs/'
obrazky = load_sudoku_images(slozka_sudoku)
priklad = obrazky[5,:,:,:]

prep_img = Preprocess(priklad)
cv2.imshow('window', prep_img)
cv2.waitKey(0)
cv2.destroyAllWindows()