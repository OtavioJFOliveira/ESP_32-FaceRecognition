import cv2
import urllib.request
import numpy as np

url='http://192.168.0.7/cam-hi.jpg'

winName = 'ESP-CAM'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)


while True:
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    cv2.imshow(winName, img)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()