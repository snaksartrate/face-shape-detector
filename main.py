import cv2 as cv

cap = cv.VideoCapture(0)   # connect to webcam (0 = default camera)

while True:
    success, frame = cap.read()   # capture one frame

    if not success:
        break

    cv.imshow("Face Shape Detector", frame)   # display frame

    if cv.waitKey(1) == ord('q'):   # press q to quit
        break

cap.release()              # free camera
cv.destroyAllWindows()    # close windows