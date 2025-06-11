import cv2 as cv

img = cv.imread('utils/img/a.jpg')
cv.imshow('Image', img)
# cv.waitKey(0)


# vid = cv.VideoCapture('utils/vid/hawk.mp4')
vid = cv.VideoCapture(0)
while True:
    isTrue, frame = vid.read()
    if not isTrue:
        break
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()

