import cv2
import numpy as np

'''水平投影'''


def cv2_show(name, img):
    cv2.namedWindow(name,0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


def GetRedresserImage():
    image = cv2.imread('image7.jpg')
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
    print(h_)
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
    cv2_show('vProjection', vProjection)

    return w_


def GetHWposition(H, w, img):
    """

    :param H: H = GetHProjection
    :param w: w = (h, w) = img.shape
    :param img: image binaire
    :return:
    """
    Position = []
    start = 0
    H_Start = []
    H_End = []
    # Obtenez la position de division verticale en fonction de la projection horizontale
    for i in range(len(H)):
        if H[i] > 0 and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    # # end of image
    # if H[len(H)-1] > 5:
    #     H_End.append(len(H)-1)

    # Séparer la ligne, puis enregistre la position de séparation
    for i in range(len(H_Start)):
        # Obtenez une projection horizontale
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        # cropImg2 = cv2.resize(cropImg, None, fx=1, fy=1)
        showAndWaitKey('cropImg', cropImg)

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
         cv2.rectangle(redresserImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)      # x1,y1 x2,y2

    cv2.imshow('redresserImage', redresserImage)
    # cv2.imshow('redresserImage*2', cv2.resize(redresserImage, None, fx=1, fy=1))
    cv2.waitKey(0)

    return Position


def ReSeparate(redresserImage, Position, img, W):

    # 找出所有混合的字母块， 存入Position2
    for i in range(len(Position)):
        distan = Position[i][2] - Position[i][0]   # x2 - x1
        if distan > 9:
            PartImg = img[Position[i][1]:Position[i][3], Position[i][0]:Position[i][2]]    # y1,y2 x1,x2
            PartReImg = redresserImage[Position[i][1]:Position[i][3], Position[i][0]:Position[i][2]]
            # montre les images à re séparer
            cv2.imshow('PartImg',  cv2.resize(PartImg, None, fx=1, fy=1))
            cv2.waitKey(0)
            # cv2.destroyWindow('PartImg')

            W = GetVProjection(PartImg)

            Position2 = []
            Wstart = 0
            Wend = 0
            W_Start = 0
            W_End = 0
            # for j in range(Position[i][0], Position[i][2]):
            for j in range(len(W)):
                if W[j] > 0 and Wstart == 0:
                    W_Start = j
                    Wstart = 1
                    Wend = 0
                if W[j] <= 0 and Wstart == 1:
                    W_End = j
                    Wstart = 0
                    Wend = 1
                if j == range(len(W)) and Wstart == 1:
                    W_End = j
                    Wstart = 0
                    Wend = 1
                if Wend == 1:
                    Position2.append([W_Start, 0, W_End, Position[i][3] - Position[i][1]])
                    print("Position2 = ", Position2)
                    print("length of Position2 = ", len(Position2))
                    Wend = 0


            #Séparer les lettres
            for m in range(len(Position2)):
                print("rectangle")
                cv2.rectangle(PartReImg, (Position2[m][0], Position2[m][1]), (Position2[m][2], Position2[m][3]), (0, 0, 255), 1)      # x1,y1 x2,y2

            cv2.imshow('PartImg twice *2', cv2.resize(PartReImg, None, fx=1, fy=1))
            print("1")
            cv2.waitKey(0)



if __name__ == "__main__":
    # read image
    redresserImage = GetRedresserImage()
    #
    image = cv2.cvtColor(redresserImage, cv2.COLOR_BGR2GRAY)
    showAndWaitKey('gray', image)
    # histogramme
    # dst = cv2.equalizeHist(image)
    # cv2_show("dst", dst)
    # image binaire
    retval, img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    showAndWaitKey('binary', img)
    (h, w) = img.shape
    image = img.copy()
    image = cv2.resize(image, None, fx=0.2, fy=0.2)
    # Projection horizontale
    H = GetHProjection(image)
    W = GetVProjection(image)

    Position = GetHWposition(H, w, img)

    # ReSeparate(redresserImage, Position, img, W)
    # W = getVProjection(img)

