import cv2
import numpy as np

'''水平投影'''


def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GetRedresserImage():
    image = cv2.imread('image5.jpg')
    cv2_show('resize', image)

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
        for x in range(h_[y]):
            hProjection[y, x] = 255
    cv2_show('hProjection2', hProjection)

    return h_


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
    # Projection horizontale
    H = getHProjection(img)

    W = getVProjection(img)

