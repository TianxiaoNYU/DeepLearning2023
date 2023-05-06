# LUAD survival analysis by using the integration of the multi-omics
Tianxiao Zhao, Ze Chen

*The project for Deep Learning in Mediceine 2023*

****

- This repo includes the codes for the whole project in the ./script folder.

  - Each sub-folder stands for the section for different task parts, which is self-descriptive.  
  - Note that all this code were run on the NYU langone BigPurple HPC. Most code-runing jobs are written as sbatch files in the slurm system. One can modify the codes to run them locally.
  - Some scripts use absolute directories. One can also modify them to make things run properly.




- Our project has three steps:

  1. image only and 1-dimensional data only

  2. Baseline Multi-omics integration

  3. Cooperative learning MO integration

Therefore, our script also has three parts.
script
   ├── 1_image

   │   ├── bashCommand.txt

   │   ├── function.py                        containing all util function

   │   ├── split_whole.image.py               cropping whole image into required size of tiles

   │   ├── step_1_download_tile.sh            step 1 download image and applying split_whole.image.py

   │   ├── step_2a_image_transformation.py    Macenko normalization

   │   ├── step_2_meta_split.py               split into training, validation, and test

   │   ├── step_3a_VGG19.py                   vgg19 on subset data

   │   ├── step_3b_inceptionV3.py             inceptionV3 on subset data

   │   ├── step_3c_resnet50_subset.py         resnet50 on subset data

   │   ├── step_4a_TZVGG19.py                 vgg19 on whole data

   │   ├── step_4c_resnet50.py                resnet50 on whole data

   │   ├── TZ_resnet50.py                     

   │   ├── TZ_resnet_test.py                  get test performance by resnet50

   │   ├── TZ_vgg_trimmed.py

   │   └── vgg_run_TZ.sh

   ├── 1_other

   │   ├── function.py                        containing all util function

   │   └── TZ_multiomics.py                   run 1dimensional model

   ├── 2_baseMulti

   │   ├── function.py                        containing all util function

   │   ├── step_1_download_1D.sh              download RNA, CNV, Meth

   │   ├── step_2_meta_split.py               split into training, validation, and test

   │   ├── step_3_1Domics.py                  scale 1D data and adding noise

   │   ├── step_4c_resnet50_subset.py         baseline_mo on subset dataset

   │   ├── step_5c_resnet.py                  baseline mo on whole data

   │   ├── TZ_basemulti_test.py               test performance from the best model

   │   └── TZ_step_4c_resnet50_subset.py      

   └── 3_coLearn               

​        ├── function.py                        containing all util function

​        ├── step_1c_resnet50_subset.py         co_mo on subset dataset

​        ├── step_2c_resnet50.py                co_mo on whole dataset

​        └── TZ_coopLearning.py                 test performance from the best model



Corresponding, our output also three parts:
├── output

│   ├── 1_image

│   │   ├── 1_image_resnet_hyperscreen_train_auc.csv

│   │   ├── 1_image_resnet_hyperscreen_valid_auc.csv

│   │   ├── 1_image_resnet_wholedata_performance_16_0.0001.png

│   │   └── 1_image_resnet_wholedata_performance_16_1e-05.png

│   ├── 2_baseMulti

│   │   ├── 2_mo_resnet_train_AUC.csv

│   │   ├── 2_mo_resnet_valid_AUC.csv

│   │   ├── 2_mo_whole_resnet_performance_16_1e-05.png

│   │   ├── 2_mo_whole_resnet_performance_16_1e-06.png

│   │   ├── Test_ROC.png

│   │   └── TZ_resnet50_res.txt

│   └── 3_coLearn

│       ├── 3_co_resnet_performance_whole_0.3_1e-05.png

│       ├── 3_co_resnet_performance_whole_0.3_1e-06.png

│       ├── 3_co_resnet_performance_whole_0.9_1e-05.png

│       ├── 3_co_resnet_performance_whole_0.9_1e-06.png

│       ├── 3_co_resnet_samllrho_train_AUC.csv

│       ├── 3_co_resnet_samllrho_valid_AUC.csv

│       ├── testing_res_coLearn_0.3_1e-6.txt

│       ├── testing_res_coLearn_0.9_1e-6.txt

│       └── Test_ROC.png
