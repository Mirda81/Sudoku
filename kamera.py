import cv2
from image_processing import Preprocess, extract_frame, Perspective_transform, extract_numbers, predict_numbers,displayNumbers, get_InvPerspective, center_numbers, get_corners

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)
# change brightness to 150
cap.set(10, 150)

while True:
    success, img = cap.read()
    img_result = img.copy()

    prep_img = Preprocess(img_result)
    frame, contour = extract_frame(prep_img)

    if len(contour) > 0:
        corners = get_corners(contour)

        for corner in corners:
            x, y = corner
            cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow('window', img_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
