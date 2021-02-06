import numpy as np
import cv2
import math


def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, wrapMat, (h, w))
    return image, wrapMat, w, h


def getCorrect():
    #
    src = cv2.imread("imagecontour.jpg", cv2.IMREAD_COLOR)
    resSrc = src.copy()
    resSrc = cv2.resize(resSrc, None, fx=0.5, fy=0.5)
    showAndWaitKey("src", resSrc)
    gray = cv2.cvtColor(resSrc, cv2.COLOR_BGR2GRAY)
    showAndWaitKey("gray", gray)
    # Corrosion, expansion
    # kernel = np.ones((5, 5), np.uint8)
    # erode_Img = cv2.erode(gray, kernel)
    # eroDil = cv2.dilate(erode_Img, kernel)
    # showAndWaitKey("eroDil", eroDil)
    # Détection de contours
    canny = cv2.Canny(gray, 50, 200)
    showAndWaitKey("canny", canny)
    # Transformation Hough pour obtenir des lignes
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=100, maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    # tracer les lignes
    print("lines = ", lines)
    mline = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mline.append(line[0])
        print("Line[0]= ", line[0])
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    showAndWaitKey("houghP", drawing)
    print("mline= ", mline)
    """
    Calculez l'angle
    """
    list_t = []
    for mmline in mline:
        x1 = mmline[0]
        y1 = mmline[1]
        x2 = mmline[2]
        y2 = mmline[3]
        if x1 == x2 or y1 == y2:
            print("x1=x2 or y1=y2")
        else:
            print("x1=", x1, "y1= ", y1)
            print("x2=", x2, "y2= ", y2)
            k = float(y1 - y2) / (x1 - x2)
            thera = np.degrees(math.atan(k))
            list_t.append(thera)
            print("thera = ", thera)

    print(list_t)

    tmoyenne = np.mean(list_t)
    print("tmoyenne = ", tmoyenne)

    """
    L'angle de rotation est supérieur à 0, puis tourner dans le sens antihoraire, sinon dans le sens horaire
    """

    rotateImg, MatRotation, w, h = rotate(src, tmoyenne)        # weigh, height

    # position of the three corner after rotate
    Q1 = np.dot(MatRotation, np.array([[h], [w], [1]]))         # x, y
    print("Q1 =", Q1)
    Q2 = np.dot(MatRotation, np.array([[0], [0], [1]]))
    print("Q2 =", Q2)
    Q3 = np.dot(MatRotation, np.array([[h], [0], [1]]))
    print("Q3 =", Q3)

    PartImg = rotateImg[int(Q2[1]):int(Q1[1]), 0:int(Q3[0])]          # y1,y2 x1,x2

    cv2.imshow("rotateImg", cv2.resize(rotateImg, None, fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.imshow("rotateImgs", cv2.resize(PartImg, None, fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.imwrite("imagerotate.jpg", PartImg)

    return PartImg


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


if __name__ == "__main__":
    p = getCorrect()
