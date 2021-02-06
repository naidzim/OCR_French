import cv2
import numpy as np
import pytesseract

def cv2_show(name, img):
    cv2.namedWindow(name,0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


def GetRedresserImage():
    image = cv2.imread('imagerotate.jpg')
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2_show('image', image)

    return image


def GetHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # Hauteur et largeur de l'image
    (h, w) = image.shape
    # Un tableau de longueur cohérent avec la hauteur de l'image
    h_ = [0] * h
    # Comptez le nombre de pixels blancs dans chaque ligne
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # image de projection horizontale
    for y in range(h):
        if h_[y] < 200:
            for x in range(h_[y]):
                hProjection[y, x] = 255
    # cv2_show('hProjection', cv2.resize(hProjection, None, fx=0.5, fy=0.5))
    showAndWaitKey('hProjection', hProjection)
    # print(h_)
    return h_


def GetVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    # Hauteur et largeur de l'image
    (h, w) = image.shape
    # Un tableau de longueur cohérent avec la largeur de l'image
    w_ = [0]*w
    # Comptez le nombre de pixels blancs dans chaque ligne
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # image de projection vertical
    for x in range(w):
        for y in range(h-w_[x], h):
            vProjection[y, x] = 255
    # print(w_)
    showAndWaitKey('vProjection', vProjection)

    return w_


def GetHWposition(H, w, img, redresserImage):
    """
    :param H: H = GetHProjection
    :param w: w = (h, w) = imagec.shape
    :param img: image binaire resize
    :return:
    """
    Position = []
    start = 0
    H_Start = []
    H_End = []
    # Obtenez la position de division verticale en fonction de la projection horizontale
    for i in range(len(H)):
        if H[i] > 10 and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] <= 10 and start == 1:
            H_End.append(i)
            start = 0

    # Séparer la ligne, puis enregistre la position de séparation
    for i in range(len(H_Start)):
        # Obtenez une projection horizontale
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        # cropImg2 = cv2.resize(cropImg, None, fx=1, fy=1)
        cv2.imshow('cropImg', cropImg)

        # Obtenez une projection vertical
        W = GetVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 0 and Wstart == 0:
                W_Start = j
                Wstart = 1
                Wend = 0
            if W[j] <= 0 and Wstart == 1:
                W_End = j
                Wstart = 0
                Wend = 1
            if Wend == 1:
                Position.append([W_Start, H_Start[i], W_End, H_End[i]])
                Wend = 0

    #Séparer les lettres
    for m in range(len(Position)):
         # cv2.rectangle(redresserImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)      # x1,y1 x2,y2
         cv2.rectangle(redresserImage, (2*Position[m][0], 2*Position[m][1]), (2*Position[m][2], 2*Position[m][3]), (0, 229, 255), 1)
         # print("Position[m][0]", Position[m][0])
         # print("Position[m][1]", Position[m][1])
         # print("Position[m][2]", Position[m][2])
         # print("Position[m][3]", Position[m][3])

    cv2.imshow('img', redresserImage)
    cv2.waitKey(0)


    img = cv2.cvtColor(redresserImage, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    showAndWaitKey('binary', img)
    dst = cv2.equalizeHist(img)
    for m in range(len(Position)):
        lettre = dst[2*Position[m][1]:2*Position[m][3], 2*Position[m][0]:2*Position[m][2]]   # [y1,y2] [x1,x2]
        lettre = cv2.resize(lettre, None, fx=2, fy=2)
        # lettre = cv2.vconcat([lettre, lettre, lettre])

        cv2.imshow('lettre', lettre)
        text = pytesseract.image_to_string(lettre, lang='fra')

        print(text)
        cv2.waitKey(0)
        cv2.destroyWindow('lettre')



if __name__ == "__main__":
    # read image
    redresserImage = GetRedresserImage()
    # GRAY
    image = cv2.cvtColor(redresserImage, cv2.COLOR_BGR2GRAY)
    showAndWaitKey('gray', image)
    # binaire
    retval, img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    showAndWaitKey('binary', img)

    imagec = img.copy()
    imagec = cv2.resize(imagec, None, fx=0.5, fy=0.5)
    # redresserImage = cv2.resize(redresserImage, None, fx=0.5, fy=0.5)
    (h, w) = imagec.shape

    # Projection
    H = GetHProjection(imagec)
    # Projection
    W = GetVProjection(imagec)

    Position = GetHWposition(H, w, imagec, redresserImage)


