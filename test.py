from PIL import Image
import os.path

filename = os.path.join("./dataset/dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg")
img = Image.open(filename)
print(img.size)