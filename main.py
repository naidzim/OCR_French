import Contour
import rotate
import projetction


import pytesseract
import cv2

# définier l'image à traiter
nomfichier = 'image4.jpg'


lettre = cv2.imread('imagerotate.jpg')
text = pytesseract.image_to_string(lettre, lang='myfra')

print(text)

# lettre = cv2.imread('imagerotate.jpg')
# cv2.imshow('lettre', lettre)
# cv2.waitKey(0)
# lettre = cv2.cvtColor(lettre, cv2.COLOR_BGR2GRAY)
# cv2.imshow('cvtColor', lettre)
# cv2.waitKey(0)
# lettre = cv2.threshold(lettre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# cv2.imshow('threshold', lettre)
# cv2.waitKey(0)
# retval, lettre = cv2.threshold(lettre, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# cv2.imshow('threshold', lettre)
# cv2.waitKey(0)
#
#
# # text = pytesseract.image_to_string(lettre, lang='fra', config="-c tessedit"
# #                                              "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# #                                              " --psm 10"
# #                                              " -l osd"
# #                                              " ")
# #

# text = pytesseract.image_to_string(lettre, lang='fra', config=" --psm 10 --oem 3")
# print(text)


if nomfichier == 'image1.png' or nomfichier == 'image2.png':
    w, imagec, ImageP = projetction.Getimage(nomfichier)
    # Projection
    H = projetction.GetHProjection(imagec)
    # Séparer les caractères
    Position = projetction.GetHWposition(H, w, imagec, ImageP)
    # reconnaissance de lettre
    projetction.Reconna1(Position, ImageP)

else:
    # localiser la partie text
    image, edged = Contour.PreTraite(nomfichier)
    # chercher le contour
    Contour.findContour(image, edged)
    # touner image de text (imagecontour.jpg)
    p = rotate.getCorrect()
    # charger l'image pour séparer les lignes et les caractères (imagerotate.jpg)
    w, imagec, ImageP = projetction.Getimage('imagerotate.jpg')
    # Projection
    H = projetction.GetHProjection(imagec)
    # Projection
    W = projetction.GetVProjection(imagec)
    # tracer les contour de chaque caractères puis faire la reconnaissance
    Position = projetction.GetHWposition(H, w, imagec, ImageP)
    # reconnaissance de lettre
    projetction.Reconna2(Position, ImageP)
