import numpy as np
import random
import openslide
import openslide.deepzoom
import os
import sys
### step 2.a claim function###
def splitImage(data_dir,
               plot_dir,
               size = 1000,
               n_subimage = 60,
               white_criteria = 650,
               white_threshold = 0.7,
               iteration = 300,
               randomseed = 20230328):
  
  random.seed(randomseed)

  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

  im = openslide.OpenSlide(data_dir)
  loc_list = []
  k = 0
  
  while k < n_subimage:
      tile_loc = (random.randrange(round(im.dimensions[0] / size) - 1) * size, 
                  random.randrange(round(im.dimensions[1] / size) - 1) * size)
      if len(loc_list) > iteration:
        break
      if tile_loc in loc_list:
        continue
      loc_list.append(tile_loc)
      res = im.read_region(tile_loc,
                           level = 0,
                           size = (size, size))
      res_color = res.quantize(colors = 2).convert('RGB').getcolors()
      white_num = []
      pixel_num = []
      for eachcolor in res_color:
        if np.sum(eachcolor[1]) > white_criteria:
          white_num.append(eachcolor[0])
        pixel_num.append(eachcolor[0])
      sum_white = np.sum(white_num) / np.sum(pixel_num)
      if sum_white < white_threshold:
        k += 1
        res.convert('RGB').save("{}/{}_{}_{}.jpg".format(plot_dir, tile_loc[0], tile_loc[1], k))
  return 0





### step 2.b run split function###
sample_folder=sys.argv[1]
plot_dir=sys.argv[2]
n_subimage = int(sys.argv[3])
image_size = int(sys.argv[4])




image_name = os.listdir( sample_folder  )
image_name = [x for x in image_name if "svs" in x][0]

#if "partial" in image_name:
#    continue
 #print(image_name)

splitImage(data_dir="{}/{}".format(sample_folder, image_name),
             plot_dir="{}/".format(plot_dir),
             size = image_size,
             n_subimage = n_subimage)




