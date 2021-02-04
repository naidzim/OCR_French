import cv2
from imutils.perspective import four_point_transform
import pytesseract
import os
from PIL import Image
import argparse
import matplotlib as plt
import numpy as np
import fonctions


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
    cv2.imwrite("image5.jpg", ref)
    print('img saved')


image = cv2.imread('image3.jpg')
resizeimg = cv2.resize(image, None, fx=0.2, fy=0.2)
cv2_show('resize', resizeimg)

#text = pytesseract.image_to_string('image3.png', lang = 'fra')
#print(text)

# pretraitement
gray = fonctions.get_grayscale(resizeimg)
cv2_show('gray', gray)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2_show('GaussianBlur', blur)
# dst = cv2.equalizeHist(gray)
# cv2_show("dst", dst)
# ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv2_show("dst", ret)
mediu = cv2.medianBlur(gray,27)
cv2_show("dst", mediu)
edged = cv2.Canny(mediu, 50, 100)
cv2_show('edged', edged)


# chercher le contour
cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# liste les contours dans l'ordre de plus large vers plus petite, on list les premières 5 contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
print('cnts', len(cnts))

i = 0

for c in cnts:
    print("i = ", i)
    i = i+1
    peri = cv2.arcLength(c, True)  # length de contour fermé
    print("length de contour = ", peri)
    approx=cv2.approxPolyDP(c, 0.02*peri, True)  # 检测出来的轮廓可能是离散的点，故因在此做近似计算，使其形成一个矩形  # True = fermé
    print("search lenapprox", len(approx))


    if len(approx) == 4:
        screenCnt = approx
        #print("lenapprox = ", approx)
        print("lenapprox", len(approx))
        # print("screenCnt = ", screenCnt)

print("screenCnt = ", screenCnt)
print("type screenCnt ", type(screenCnt))
# contour 1, les cordonnées des qutres points (921, 300), (662,49), (0,139), (49,800)
# point_size = 1
# point_color = (255, 255, 0) # BGR
# thickness = 4 # 可以为 0 、4、8
# points_list = [(921, 300), (662,49), (0,139), (49,800)]

# for point in points_list:
#     cv2.circle(edged, point, point_size, point_color, thickness)
# cv2_show('point', edged)

if screenCnt.any() == 0:
    print("no contour")
else:
    print("there r contour")
    image = resizeimg.copy()
    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)  # tracer les contours，-1 répresente tracer tous les contours
    cv2_show('contour', image)
    # redresser l'image
    wraped = four_point_transform(image, screenCnt.reshape(4, 2))
    wraped = cv2.resize(wraped, None, fx=0.5, fy=0.5)
    cv2_show('wrap', wraped)
    #rr(wraped)

