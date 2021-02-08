import cv2
import numpy as np
import pytesseract


def Getimage(ImageP):
    # read image
    ImageP = cv2.imread(ImageP)
    ImageP = cv2.resize(ImageP, None, fx=0.5, fy=0.5)
    # cv2_show('ImageP', ImageP)
    # GRAY
    image = cv2.cvtColor(ImageP, cv2.COLOR_BGR2GRAY)
    # showAndWaitKey('gray', image)
    # binaire
    retval, img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # showAndWaitKey('binary', img)

    imagec = img.copy()
    imagec = cv2.resize(imagec, None, fx=0.5, fy=0.5)
    (h, w) = imagec.shape

    return w, imagec, ImageP


def cv2_show(name, img):
    cv2.namedWindow(name,0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


def GetHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # Hauteur et largeur de l'image
    (h, w) = image.shape
    # Un tableau de longueur cohérent avec la hauteur de l'image
    # pour stocker les pixels blancs dans chaque lignes
    h_ = [0] * h
    # Comptez le nombre de pixels blancs dans chaque ligne
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # image de projection horizontale (Debug)
    # for y in range(h):
    #     if h_[y] < 200:
    #     for x in range(h_[y]):
    #         hProjection[y, x] = 255
    # showAndWaitKey('hProjection', hProjection)
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
    # for x in range(w):
    #     for y in range(h-w_[x], h):
    #         vProjection[y, x] = 255
    # print(w_)
    # showAndWaitKey('vProjection', vProjection)

    return w_


def GetHWposition(H, w, img, ImageP):
    """
    :param H: H = GetHProjection
    :param w: w = (h, w) = imagec.shape
    :param img: image binaire resize
    :param ImageP:
    :return:
    """
    positionMots = []  # Position des mots
    positionChar = []
    start = 0
    H_Start = []       # debuts des lignes
    H_End = []         # fins de lignes

    # stocker les positions horizontales de debut et de fin de chaque ligne de texte
    for i in range(len(H)):
        if H[i] > 10 and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] <= 10 and start == 1:
            H_End.append(i)
            start = 0

    # Séparer la ligne, puis enregistre la position de séparation
    for i in range(len(H_Start)):
        #projection horizontale pour separer les mots
        lineImg = img[H_Start[i]:H_End[i], 0:w] #image de chaque ligne de texte
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        lineImgDilated = cv2.dilate(lineImg, kernel)

        # projection vertical pour separer les mots des lignes
        W = GetVProjection(lineImgDilated)
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
                positionMots.append([W_Start, H_Start[i], W_End, H_End[i]])
                Wend = 0

    # projection vertical pour separer les characteres des mots

    for i in range(len(positionMots)):
        #projection horizontale pour separer les mots
        motImg = img[positionMots[i][1]:positionMots[i][3], positionMots[i][0]:positionMots[i][2]] #image de chaque ligne de texte
        # cv2.imshow('motImg1', motImg)
        # cv2.waitKey(0)
        # # cv2.destroyWindow('charImg')

        # projection vertical pour separer les characteres des mots
        W = GetVProjection(motImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0

        positionCharInt = []
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
                positionCharInt.append([W_Start, positionMots[i][1], W_End, positionMots[i][3]])
                Wend = 0
        positionChar.append(positionCharInt)
    # Séparer les chars (pour affichage)
    for i in range(len(positionMots)) :
        motImg = img[positionMots[i][1]:positionMots[i][3], positionMots[i][0]:positionMots[i][2]]

        # cv2.imshow('motImg2', motImg)
        # cv2.waitKey(0)
        # for m in range(len(positionChar[i])):
        #     print(positionChar[m])
        #     myImg = motImg[positionChar[i][m][1]:positionChar[i][m][3], positionChar[i][m][0]:positionChar[i][m][2]]
        #     print(motImg.shape)
        #     print(myImg.shape)
        #     # cv2.imshow('charImg', myImg)
        #     # cv2.waitKey(0)
        #     # cv2.destroyWindow('charImg')
        # cv2.imshow('charImg', motImg)
        # cv2.waitKey(0)
        # cv2.destroyWindow('charImg')

    #Séparer les mots (pour affichage)
    # for m in range(len(positionMots)):
    #      cv2.rectangle(ImageP, (2*positionMots[m][0], 2*positionMots[m][1]),
    #                    (2*positionMots[m][2], 2*positionMots[m][3]), (0, 229, 255), 1)


    for i in range(len(positionChar)):
        print(positionChar[i])
    # cv2.imshow('img', ImageP)
    # cv2.waitKey(0)
    return positionMots, positionChar

def getLigne(img) :
    positionLignes = []
    start = 0
    H = GetHProjection(img)
    ligneDebut = []
    ligneFin = []

    # stocker les positions horizontales de debut et de fin de chaque ligne de texte

    for i in range(len(H)):
        if H[i] > 10 and start ==0:
            ligneDebut.append(i)
            start = 1
        if H[i] <= 10 and start == 1:
            ligneFin.append(i)
            start = 0

    for i in range(len(ligneDebut)):
        # print("positionLignes = (%d, %d)" % (ligneDebut[i], ligneFin[i]))
        positionLignes.append([ligneDebut[i], ligneFin[i]])
    # print("positionLignes = ", positionLignes)

    """Debug"""
    # for i in range(len(positionLignes)):
    #     print(positionLignes[i])
    #     lineImg = img[positionLignes[i][0]:positionLignes[i][1], 0:w]  # image de chaque ligne de texte
    #     cv2.imshow("linge",lineImg)
    #     cv2.waitKey()

    return positionLignes

def getMots(img,positionLignes):

    positionMots=[]

    (h, w) = img.shape # dimension de l'image (height, width)
    # print (positionLignes )
    # for i in range(len(positionLignes)):
        #projection horizontale pour separer les mots
    lineImg = img[positionLignes[0]:positionLignes[1], 0:w]  # image de chaque ligne de texte    # [y1,y2] [x1,x2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    lineImgDilated = cv2.dilate(lineImg, kernel)

    # projection vertical pour separer les mots des lignes
    W = GetVProjection(lineImgDilated)
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
            positionMots.append([W_Start, W_End, positionLignes[0], positionLignes[1]])               # x1,x2 y1,y2
            Wend = 0

    # """Debug"""
    # for i in range(len(positionMots)):
    #     motImg = img[positionMots[i][2]:positionMots[i][3],positionMots[i][0]:positionMots[i][1]]               # [y1,y2] [x1,x2]
    #     print("positionMots[i][2] = ", positionMots[i][2])
    #     print("positionMots[i][3] = ", positionMots[i][3])
    #     print("positionMots[i][0] = ", positionMots[i][0])
    #     print("positionMots[i][1] = ", positionMots[i][1])
    #     # cv2.imshow('motImg', motImg)
    #     # cv2.waitKey()
    #     # cv2.destroyWindow('motImg')

    return positionMots


def getCharactere(img, positionMots):

    positionChar = []

    motImg = img[positionMots[2]:positionMots[3], positionMots[0]:positionMots[1]]           # [y1,y2] [x1,x2]
    (h, w) = motImg.shape
    W = GetVProjection(motImg)
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
            positionChar.append([W_Start, W_End, 0, h])
                                    # positionMots[i][2], positionMots[i][3]])        # x1,x2 y1,y2
            Wend = 0

    mot = ''
    for j in range(len(positionChar)):
            # charImg = motImg[positionChar[i][j][2]:positionChar[i][j][3], positionChar[i][j][0]:positionChar[i][j][1]]      # [y1,y2] [x1,x2]
            charImg = motImg[0:h+3, positionChar[j][0]-3:positionChar[j][1]+2]  # [y1,y2] [x1,x2]
            #cv2.threshold(charImg, 0, 255, cv2.THRESH_BINARY_INV)
            charImg = cv2.resize(charImg,None,fx=6, fy=6 )

            thresh = cv2.threshold(charImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # blur = cv2.GaussianBlur(charImg, (7, 7), 0)
            result = 255- thresh
            # cv2.imshow('charImg', blur)

            # cv2.imshow('thresh', thresh)
            # cv2.imshow('result', result)

            text = pytesseract.image_to_string(result, lang='fra', config=" --psm 13 ")

            string = text[0:len(text) - 2]
            # print(string, end='')
            mot = mot + string
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    return mot

def imageToText(img):
    text=''
    positionLignes = getLigne(img)
    for i in range(len(positionLignes)) : #nb de ligne
        positionMots = getMots(img,positionLignes[i])
        for j in range (len(positionMots)) : #nb de mots dans chaque ligne
            mot = getCharactere(img,positionMots[j])
            text += mot + ' '
        text += "\n"
    print(text)
    return text

def Reconna1(Position, ImageP):
    img = cv2.cvtColor(ImageP, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    showAndWaitKey('binary', img)
    dst = cv2.equalizeHist(img)

    # for m in range(len(Position)):
    #     lettre = dst[2*Position[m][1]-5:2*Position[m][3]+5, 2*Position[m][0]-5:2*Position[m][2]+5]   # [y1,y2] [x1,x2]
    #     lettre = cv2.resize(lettre, None, fx=2, fy=2)
    #     if m == 0:
    #         llettre = lettre
    #     else:
    #         llettre = cv2.hconcat([llettre, lettre])

    # cv2.imshow('llettre', llettre)
    cv2.imshow('lettre', dst)
    text = pytesseract.image_to_string(dst, lang='fra')
    print(text)
    cv2.waitKey(0)
    cv2.destroyWindow('lettre')


def Reconna2(positionMots, positionChar, ImageP):
    img = cv2.cvtColor(ImageP, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    showAndWaitKey('binary', img)
    dstImg = cv2.equalizeHist(img)
    # dstImg = cv2.resize(dstImg, None, fx=0.5, fy=0.5)
    for m in range(len(positionMots)):
        mot = dstImg[2*positionMots[m][1]-2:2*positionMots[m][3]+2, 2*positionMots[m][0]-2:2*positionMots[m][2]+2]   # [y1,y2] [x1,x2]
        mot = cv2.resize(mot, None, fx=2, fy=2)
        cv2.imshow('mot', mot)
        for n in range(len(positionChar[m])):
            char = mot[2*positionChar[m][n][1]-2:2*positionChar[m][n][3]+2, 2*positionChar[m][n][0]-2:2*positionChar[m][n][2]+2 ]
            print("positionChar[m][n][1] = ",2*positionChar[m][n][1]-2)
            print("positionChar[m][n][3] = ", 2*positionChar[m][n][3]+2)
            print("positionChar[m][n][0] = ", 2*positionChar[m][n][0]-2)
            print("positionChar[m][n][2] = ", 2*positionChar[m][n][2]+2)
            char = cv2.resize(char, None, fx=2, fy=2)


            # cv2.imshow('char', char)
            # cv2.waitKey(0)
            # cv2.destroyWindow('char')

            text = pytesseract.image_to_string(char, lang='fra', config=" --psm 10 --oem 3")
            string = text[0:len(text)-1]
            print (string,end='')
            cv2.waitKey(0)
            cv2.destroyWindow('mot')


if __name__ == "__main__":
    imageP = 'imagerotate.jpg'
    w, imagec, ImageP = Getimage(imageP)
    # Projection
    # H = GetHProjection(imagec)
    # Projection
    # W = GetVProjection(imagec)

    # positionLignes = getLigne(imagec)
    # positionMots = getMots(imagec,positionLignes)
    # getCharactere(imagec,positionMots)
    text = imageToText(imagec)
    #positionMots, positionChar  = GetHWposition(H, w, imagec, ImageP)
    # Reconna2(positionMots, positionChar, ImageP)


