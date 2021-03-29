import cv2 as cv

def goLangVersion(coordinate, image) : 
    x = int(coordinate[0])
    y = int(coordinate[1])

    img = cv.imread(image.name)
    width = img.shape[1]
    height = img.shape[0]

    a = y*width + x
    b = width*height + y*width + x
    c = 2*width*height + y*width + x

    return a, b, c

def jiaChenVersion(coordinate, image) : 
    x = int(coordinate[0])
    y = int(coordinate[1])

    img = cv.imread(image.name)
    pixelValue = img[x][y] / 255

    return pixelValue