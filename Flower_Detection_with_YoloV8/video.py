from ultralytics import YOLO
import cv2
import sys

model = YOLO("best (2).pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera Failed")
    sys.exit()

while True:
    ret, image = camera.read()
    if not ret:
        break
    pred = model.predict(image, show=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
