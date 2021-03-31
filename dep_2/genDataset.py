import os
import argparse

from PIL import Image
from preProcess import calibrateImages, loadImages

parser = argparse.ArgumentParser() 
parser.add_argument('-f', '--filename', type=str, default="dino_par.txt", help="The name of the file containing the name of the images and their corresponding camera parameters.")
parser.add_argument('-d', '--dirname', type=str, default="C:/Users/emilt/multiview-reconstruction/src/data/sample04/dino/dino", help="The directory contain the input file.")
args = parser.parse_args()
os.chdir(args.dirname)

images = loadImages(args.filename)
calibrateImages(images)
id = 0
for image in images : 
    name = f'00000{id:03d}'
    pre1 = "visualize/"
    pre2 = "txt/"
    ext1 = ".txt"
    ext2 = ".jpg"
    img = Image.open(image.name) 
    jpg = img.convert('RGB')
    jpg.save(pre1 + name + ext2)
    filename = pre2 + name + ext1
    pMat = image.projectionMatrix
    file = open(filename, "a")
    file.write("CONTOUR")
    file.write("\n")
    file.write(str(pMat[0][0]))
    file.write(" ")
    file.write(str(pMat[0][1]))
    file.write(" ")
    file.write(str(pMat[0][2]))
    file.write(" ")
    file.write(str(pMat[0][3]))
    file.write("\n")
    file.write(str(pMat[1][0]))
    file.write(" ")
    file.write(str(pMat[1][1]))
    file.write(" ")
    file.write(str(pMat[1][2]))
    file.write(" ")
    file.write(str(pMat[1][3]))
    file.write("\n")
    file.write(str(pMat[2][0]))
    file.write(" ")
    file.write(str(pMat[2][1]))
    file.write(" ")
    file.write(str(pMat[2][2]))
    file.write(" ")
    file.write(str(pMat[2][3]))
    file.close()
    id += 1 