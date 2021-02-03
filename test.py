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
    cv2.imwrite("F:/image3.jpg", ref)
    print('img write')


image = cv2.imread('image3.jpg')
resizeimg = cv2.resize(image, None, fx=0.2, fy=0.2)
cv2_show('resize', resizeimg)

# detecter avec langue français
#text = pytesseract.image_to_string('image3.png', lang = 'fra')
#print(text)

# pretraitement
gray = fonctions.get_grayscale(resizeimg)
cv2_show('gray', gray)
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2_show('GaussianBlur', blur)
# dst = cv2.equalizeHist(gray)
# cv2_show("dst", dst)
mediu = cv2.medianBlur(gray, 41)
cv2_show("dst", mediu)
edged = cv2.Canny(mediu, 50, 100)
cv2_show('edged', edged)

# HoughLinesP ne marche pas dans notre cas
# 画出直线，画出屏幕四周的线，相交可得出四个顶点。 画不出来 而且屏幕边框的线并不平行，
# minLineLength = 100
# maxLineGap = 30
# lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 200, minLineLength, maxLineGap)
#
# print("line = ", lines)
# print("length lines = ", len(lines))
# length = len(lines)
# for i in range(len(lines)):
#     for x1, y1, x2, y2 in lines[i]:
#         cv2.line(resizeimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# cv2_show("houghline", resizeimg)



# chercher le contour

# trouver les coutours
cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# liste les contours dans l'ordre de plus large vers plus petite, on list les premières 5 contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
print('cnts', len(cnts))

approx=cv2.approxPolyDP(cnts[0], 10, True)  # cnts[0] = le coutour le plus large , True = contour fermé
print("search lenapprox", len(approx))
print("approx = ", approx)
# points = fonctions.my_point(approx[0][0][0], approx[0][0][1])
# print(points)

# lister tous les points
points = []
for j in range(len(approx)):
    npoints = fonctions.my_point(approx[j][0][0], approx[j][0][1])
    points.append(npoints)
#    print(points)

four_points = points[:]     # copier la tableau

# calculer les distances entre les points
for j in range(len(points)):
    print("j =", j)
    for k in range(len(points[j+1:])):
        print("k =", k)
        # print(points[j], points[j+k+1])
        d = fonctions.getDistance(points[j], points[j+k+1])
        # print("d = ", d)
        if d < 200:
            # print("point in points = ", points[j+k+1])
            # print("remove points", four_points[j+k+1], "number = ", j+k+1)
            four_points.pop(j+k+1)                           # remove with index
            four_points.insert(j+k+1, (0, 0))
            # print("temp = ", four_points)


print("four_points= ", four_points)
print("points= ", points)

# remove all (0,0)
for i in range(len(four_points) - 4):
    four_points.remove((0, 0))
print("four_points= ", four_points)


screenCnt = np.array(four_points)
print("screenCnt= ", screenCnt)
image = resizeimg.copy()
cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)  # tracer les contours，-1 répresente tracer tous les contours
cv2_show('contour', image)
# redresser l'image
wraped = four_point_transform(image, screenCnt.reshape(4, 2))
wraped = cv2.resize(wraped, None, fx=0.5, fy=0.5)
cv2_show('wrap', wraped)
