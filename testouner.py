import numpy as np
import cv2
import math


def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


# 使用霍夫变换
def getCorrect():
    # 读取图片，灰度化
    src = cv2.imread("image5.jpg", cv2.IMREAD_COLOR)
    showAndWaitKey("src", src)
    gray = cv2.imread("image5.jpg", cv2.IMREAD_GRAYSCALE)
    showAndWaitKey("gray", gray)
    # 腐蚀、膨胀
    kernel = np.ones((5, 5), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)
    showAndWaitKey("eroDil", eroDil)
    # 边缘检测
    canny = cv2.Canny(eroDil, 50, 150)
    showAndWaitKey("canny", canny)
    # 霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    # 画出线条
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
    计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
    """
    list_t = []
    for mmline in mline:
        x1= mmline[0]
        y1= mmline[1]
        x2= mmline[2]
        y2= mmline[3]
        if x1 == x2 or y1 == y2 :
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
    旋转角度大于0，则逆时针旋转，否则顺时针旋转
    """
    rotateImg = rotate(src, list_t[1])
    cv2.imshow("rotateImg", rotateImg)
    cv2.waitKey()

    return rotateImg


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()

def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    showAndWaitKey('vProjection', vProjection)
    return w_


if __name__ == "__main__":
    r = getCorrect()
    cv2.imwrite("image6.jpg", r)
    m = cv2.imread("imagel.jpg")
    showAndWaitKey("m", m)
    getVProjection(m)