import cv2 as cv

def change_resolution(vid, width, height):
    vid.set(3, width)
    vid.set(4, height)

vid = cv.VideoCapture(0)
change_resolution(vid, 680, 480) 

while True:
    isTrue, frame = vid.read()
    if not isTrue:
        break

    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
