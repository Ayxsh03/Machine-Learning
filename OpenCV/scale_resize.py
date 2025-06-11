import cv2 as cv

def rescale_frame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('utils/img/a.jpg')
resized_img = rescale_frame(img, scale=0.2)
cv.imshow('Resized Image', resized_img)
# cv.waitKey(0)

vid = cv.VideoCapture('utils/vid/hawk.mp4')
while True:
    isTrue, frame = vid.read()
    if not isTrue:
        break

    resized_frame = rescale_frame(frame, scale=0.2)
    cv.imshow('Resized Video', resized_frame)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()