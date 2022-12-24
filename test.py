# import cv2
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import numpy as np

a = np.array([[1, 9], [3, 4, 5], [1, 3, 4, 9], 2, 8, [1, 3, 6, 9], [1, 6, 9], [3, 4, 6], 7],dtype='object')
print(a[[0,1,2]])