import cv2
import os

output_folder = "captured_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture('test.mp4')
#cap = cv2.VideoCapture(0) # webcam
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("End of video or cannot read the video.")
        break
    cv2.imshow('window-name',frame)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        file_name = os.path.join(output_folder, f"face_{count}.jpg")
        cv2.imwrite(file_name, face)
        count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
       break

print(f"Total faces captured: {count}")
cap.release()
cv2.destroyAllWindows()