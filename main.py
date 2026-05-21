import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)   # connect to webcam (0 = default camera)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    success, frame = cap.read()   # capture one frame

    if not success:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv.imshow("Face Shape Detector", frame)   # display frame

    if cv.waitKey(1) == ord('q'):   # press q to quit
        break

cap.release()              # free camera
cv.destroyAllWindows()    # close windows