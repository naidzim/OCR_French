import cv2
import pytesseract

def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('image1.png')
resizeimg = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2_show('resize', resizeimg)

text = pytesseract.image_to_string(image, lang='fra')  # fra = français
print(text)
