import cv2
from imutils.perspective import four_point_transform
import pytesseract
import os
from PIL import Image
import argparse
import matplotlib as plt
import numpy as np
import fonctions

#text = pytesseract.image_to_string('image3.png', lang = 'fra')
#print(text)


def cv2_show(name, img):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('image3.jpg')

cv2_show('image', image)

# pretraitement
gray = fonctions.get_grayscale(image)
cv2_show('gray', gray)

mediu = cv2.medianBlur(gray, 91)
cv2_show("dst", mediu)

retval, threshold = cv2.threshold(mediu, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
cv2_show('threshold', threshold)

edged = cv2.Canny(threshold, 75, 200)
cv2_show('edged', edged)


# chercher le contour
cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# liste les contours dans l'ordre de plus large vers plus petite, on list les premières 5 contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
print('cnts', len(cnts))

i = 0
flag = 0
for c in cnts:
    print("i = ", i)
    i = i+1
    peri = cv2.arcLength(c, True)  # length de contour fermé
    print("length de contour = ", peri)
    approx=cv2.approxPolyDP(c, 0.02*peri, True)  # True = fermé
    print("search lenapprox", len(approx))

    if len(approx) == 4:
        screenCnt = approx
        break
    else:
        flag = 1
        screenCnt = None

print("screenCnt = ", screenCnt)
print("type screenCnt ", type(screenCnt))

if flag == 1:
    print("no contour")
else:
    print("there r contour")
    image2 = image.copy()
    cv2.drawContours(image2, [screenCnt], -1, (0, 0, 255), 2)  # tracer les contours，-1 répresente tracer tous les contours
    cv2_show('contour', image2)
    # redresser l'image
    wraped = four_point_transform(image2, screenCnt.reshape(4, 2))
    cwraped = wraped.copy()
    cwraped = cv2.resize(cwraped, None, fx=0.5, fy=0.5)
    cv2_show('wrap', cwraped)

    cv2.imwrite("imagecontour.jpg", wraped)
    print('img saved')
