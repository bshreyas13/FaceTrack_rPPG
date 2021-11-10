import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture('s01_trial16.avi')

while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape
    print("Video On")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_image)
    
    print(result.multi_face_landmarks)
    for facial_landmarks in result.multi_face_landmarks:
    
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
        h, w, c = image.shape
        cx_min=  w
        cy_min = h
        cx_max= cy_max= 0
        for id, lm in enumerate(facial_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if cx<cx_min:
                cx_min=cx
            if cy<cy_min:
                cy_min=cy
            if cx>cx_max:
                cx_max=cx
            if cy>cy_max:
                cy_max=cy
        cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
            #cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

    cv2.imshow("Image", image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
