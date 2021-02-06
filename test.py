import cv2
from imutils.perspective import four_point_transform
import pytesseract
import numpy as np
import fonctions

# définir l'image à traiter
TestImage = 'image3.jpg'


def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GetResizeImg():
    image = cv2.imread(TestImage)
    resizeimg = cv2.resize(image, None, fx=0.2, fy=0.2)
    cv2_show('resize', resizeimg)

    return resizeimg


def PreTraitement(resizeimg):
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

    return edged


def FindContour(edged):
    """
    chercher les coutours
    :param edged:
    :return:
    """
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
        # print("j =", j)
        for k in range(len(points[j+1:])):
            # print("k =", k)
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

    return four_points


def Redresser(resizeimg, four_points):
    coordonne = np.array(four_points)
    print("coordonne= ", coordonne)
    image = resizeimg.copy()
    cv2.drawContours(image, [coordonne], -1, (0, 0, 255), 2)  # tracer les contours，-1 répresente tracer tous les contours
    cv2_show('contour', image)

    # redresser l'image
    wraped = four_point_transform(image, coordonne.reshape(4, 2))
    wraped = cv2.resize(wraped, None, fx=1, fy=1)
    cv2_show('wrap', wraped)
    cv2.imwrite("image5test.jpg", wraped)
    return wraped


def GetImgDetecter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2_show('gray', image)
    # image binaire
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2_show('binary', img)

    return img


def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # Hauteur et largeur de l'image
    (h, w) = image.shape
    # Un tableau de longueur cohérent avec la hauteur de l'image
    h_ = [0] * h
    # Comptez le nombre de pixels blancs dans chaque ligne
    text_x = []
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # image de projection horizontale
    for y in range(h):
        print("length h_y = ", h_[y])
        for x in range(h_[y]):
            hProjection[y, x] = 255

    cv2_show('hProjection', hProjection)

    return h_



# def rr(wraped):
#     """
#     redresser l'image
#     test.image est l'image pour faire la suite (reconnaissance des caractères)
#     :param wraped:
#     :return:
#     """
#     print('1')
#     BiImage = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
#     print('2')
#     cv2_show('BiImage', BiImage)
#     print('3')
#     retval, img = cv2.threshold(BiImage, 127, 255, cv2.THRESH_BINARY_INV)
#     print('4')
#     cv2_show('ref', img)
#     print('5')
#     cv2.imwrite("image5.jpg", img)
#     print('img saved')


def Lecture(wraped):
    # detecter avec langue français
    text = pytesseract.image_to_string(wraped, lang = 'fra')
    print(text)


if __name__ == '__main__':

    image = GetResizeImg()

    edged = PreTraitement(image)

    FourPoints = FindContour(edged)

    wraped = Redresser(image, FourPoints)

    Wimage = GetImgDetecter(wraped)

    H = getHProjection(Wimage)

