import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

# Draw a rectangle
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)
# Draw a circle
cv.circle(blank, (250, 250), 40, (255, 0, 0), thickness=2)
# Draw a line
cv.line(blank, (0, 0), (250, 250), (0, 0, 255), thickness=2)
# Draw a text
cv.putText(blank, 'OpenCV', (300, 250), 
           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

cv.imshow('Shapes', blank)
cv.waitKey(0)
cv.destroyAllWindows()