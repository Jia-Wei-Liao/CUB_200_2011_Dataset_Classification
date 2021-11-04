# VRDL_HW1_CUB_200_2011_Dataset_Classification
## Environment
numpy==1.19.3
torch==1.9.0+cu111
torchvision==0.10.0+cu111


## File Structure
      .
      ├──data
         ├──training_images
            ├──XXXX.jpg
         ├──testing_images
            ├──XXXX.jpg
         ├──classes.txt           # contain 200 bird species and it's number (e.g. 001.Black_footed_Albatross)
         ├──training_labels.txt   # filename and label mapping (e.g. 4283.jpg 115.Brewer_Sparrow)
         ├──testing_img_order.txt # test filename (e.g. 4282.jpg)
         ├──fold                  # we split training data into 5 fold (i=1,2,3,4,5)
             ├──fold1.txt
             ├──fold2.txt
             ├──fold3.txt
             ├──fold4.txt
             ├──fold5.txt
      ├──src                      # functions inside
      ├──checkpoint               # trained model weights (you need to download this file on https://drive.google.com/drive/folders/10Lt3mBJ0sucQNb-WwA1Llq-XLHWpjfVn?usp=sharing
      ├──main01_training.py
      ├──main02_ensemble.py
      └──README.md


## Training
`main01_training.py`


## Inferece
`main02_ensemble.py`