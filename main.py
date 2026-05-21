import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)   # connect to webcam (0 = default camera)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    success, frame = cap.read()   # capture one frame

    if not success:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmakrs in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmakrs, mp_face_mesh.FACEMESH_TESSELATION)

    cv.imshow("Face Shape Detector", frame)   # display frame

    if cv.waitKey(1) == ord('q'):   # press q to quit
        break

cap.release()              # free camera
cv.destroyAllWindows()    # close windows