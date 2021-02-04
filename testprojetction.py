import cv2
import numpy as np

'''水平投影'''


def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GetRedresserImage():
    image = cv2.imread('image6.jpg')
    cv2_show('image', image)

    return image


def getHProjection(image):
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
    cv2_show('hProjection2', hProjection)
    print(h_)
    return h_

def getHWposition(H, w):
    Position = []
    start = 0
    H_Start = []
    H_End = []
    #根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 10 and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] <= 10 and start == 1:
            H_End.append(i)
            start = 0
    # end of image
    if H[len(H)-1] > 5:
        H_End.append(len(H)-1)
    #分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        #获取行图像
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        cropImg = cv2.resize(cropImg, None, fx=2, fy=2)
        cv2_show('cropImg', cropImg)

        # 对行图像进行垂直投影
        W = getVProjection(cropImg)
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
    #根据确定的位置分割字符
    for m in range(len(Position)):
        cv2.rectangle(origineImage, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (0 ,229 ,238), 1)
    cv2.imshow('image',origineImage)
    cv2.waitKey(0)


def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    # Hauteur et largeur de l'image
    (h,w) = image.shape
    # Un tableau de longueur cohérent avec la largeur de l'image
    w_ = [0]*w
    # Comptez le nombre de pixels blancs dans chaque ligne
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    # image de projection vertical
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    cv2_show('vProjection', vProjection)
    print(w_)
    return w_


if __name__ == "__main__":
    # read image
    origineImage = GetRedresserImage()
    #
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    cv2_show('gray', image)
    # image binaire
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2_show('binary', img)
    (h, w) = img.shape
    # Projection horizontale
    H = getHProjection(img)

    getHWposition(H, w)

    #W = getVProjection(img)

