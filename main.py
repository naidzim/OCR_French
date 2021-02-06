import Contour
import rotate
import projetction

# définier l'image à traiter
nomfichier = 'image3.jpg'


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
