import cv2
import urllib.request
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

url = 'http://192.168.0.7/cam-hi.jpg'

winName = 'ESP-CAM'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)


def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)

    x1, y1, width, height = box

    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)


while True:
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    frame = img

    faces = detector.detect_faces(frame)

    for face in faces:
        confidence = face['confidence'] * 100

        if confidence >= 98:
            x1, y1, w, h = face['box']

            color = (192, 255, 119)

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            cv2.putText(frame, "", (x1, y1 - 10), font, fontScale=font_scale, color=color, thickness=1)

    cv2.imshow("Face Identification", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

#cap.release()
cv2.destroyAllWindows()