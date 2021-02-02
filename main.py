import cv2
from imutils.perspective import four_point_transform
import pytesseract
import os
from PIL import Image
import argparse
import matplotlib as plt
import numpy as np

def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rr(wraped):
    """
    redresser l'image

    test.image est l'image pour faire la suite (reconnaissance des caractères)
    :param wraped:
    :return:
    """
    wraped = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
    cv2_show('wrap2',wraped)
    ref = cv2.threshold(wraped, 100, 255, cv2.THRESH_BINARY)[1]
    cv2_show('ref',ref)
    ref=cv2.resize(ref,None,fx=0.5,fy=0.5)
    cv2.imwrite("F:/image3.jpg", ref)


image = cv2.imread('image3.jpg')
resizeimg = cv2.resize(image, None, fx=0.2, fy=0.2)
cv2_show('resize', resizeimg)

#text = pytesseract.image_to_string('image3.png', lang = 'fra')
#print(text)

# pretraitement
gray = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2GRAY)
cv2_show('gray', gray)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2_show('GaussianBlur', blur)
edged = cv2.Canny(blur, 75, 200)
cv2_show('edged', edged)


# chercher le contour
cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# liste les contours dans l'ordre de plus large vers plus petite, on list les premières 5 contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
print('cnts',len(cnts))
i = 0
for c in cnts:
    print("i = ", i)
    i = i+1
    peri = cv2.arcLength(c, True)  # length de contour fermé
    print("length de contour = ", peri)
    approx=cv2.approxPolyDP(c, 0.02*peri, True)  # 检测出来的轮廓可能是离散的点，故因在此做近似计算，使其形成一个矩形
    # 做精度控制，原始轮廓到近似轮廓的最大的距离，较小时可能为多边形；较大时可能为矩形
    # True表示闭合
    # print(approx)
    print("search lenapprox", len(approx))
    screenCnt = 0
    if len(approx) == 4:
        screenCnt = approx
        print("lenapprox", len(approx))
        # print("screenCnt = ", screenCnt)

if screenCnt == 0:
    print("no contour")
else:
    image = resizeimg.copy()
    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)  # tracer les contours，-1 répresente tracer tous les contours
    cv2_show('contour', image)
    # redresser l'image
    wraped = four_point_transform(image, screenCnt.reshape(4, 2)*10)
    wraped = cv2.resize(wraped, None, fx=0.2, fy=0.2)
    cv2_show('wrap', wraped)
    rr(wraped)

