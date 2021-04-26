from PIL import Image
import os

path = 'data/mydata2/img/'

i = 0
for file in os.listdir(path):
    current = os.path.join(path, file)
    if os.path.isfile(current):
        print(current)
        im = Image.open(current)
        rgb_im = im.convert('RGB')
        rgb_im.save(path + f'000000{i:02d}.jpg')
        i += 1