import sys
sys.path.append("./script/1_image")
from function import *
import pathlib 
import re
import cv2
sample = pathlib.Path("/gpfs/home/chenz05/DL2023/input/image")
sample = list(sample.iterdir())
image_path = [ list(x.iterdir()) for x in sample] 
image_path = [x for p in image_path for x in p  ] 

existing_sample = pathlib.Path("/gpfs/home/chenz05/DL2023/input/image_norm")
existing_sample = list( existing_sample.iterdir())
existing_image_path = [ list(x.iterdir()) for x in existing_sample] 
existing_image_path = [x for p in existing_image_path for x in p  ]

for i in image_path[5153:]:
  if i in existing_image_path:
     print(i," has been normalized ")
  else:
     img = io.imread(i)
     img = normalizeStaining(img,HE_ref)
     new_path = re.sub("image", "image_norm",str(i))
     cv2.imwrite(new_path, img)

