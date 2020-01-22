import cv2
import numpy
import time

from comtypes.automation import _


def detect_weed():
    camera = cv2.VideoCapture(0)
    print("Taking image...")
    img = camera.read()
    # img="C:\my_app\ImageDetection\weed_detection_algorithms\Automatic-Weed-Detection-master\\13381168_882325588543265_2073766996_n.jpg"
    if _ is True:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,(36, 0, 0), (86, 255, 255) )
        #cv2.imshow("green",mask)
        kernel = numpy.ones((5,5), numpy.uint8)
        img_erosion = cv2.erode(mask, kernel, iterations=1)
        #cv2.imshow("green",img_erosion)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        n_white_pix = numpy.sum(img_dilation == 255)
        print('Number of white pixels:', n_white_pix)
        if n_white_pix >= 100000:
            asd=1
        else:
            asd=0
        return asd
    return 0

if __name__ == '__main__':
    detect_weed()