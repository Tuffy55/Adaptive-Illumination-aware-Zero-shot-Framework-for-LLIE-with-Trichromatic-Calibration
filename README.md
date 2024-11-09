# Adaptive Illumination-aware Zero-shot Framework for Low-light Image Enhancement with Trichromatic Calibration



## Requirements
1. Python 3.8 
2. Pytorch 2.0.0
3. opencv
4. torchvision 0.15.1
5. cuda 10.1



### Folder structure
```
├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like DarkFace and ExDark.
│   │   ├── darkface 
│   │   └── exdark
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # network
├── dataloader.py
├── snapshots
│   ├── Epoch19.pth #  A pre-trained snapshot (Epoch19.pth)
```
### Test: 
```
python code/lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
Download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> and unzip and put the  downloaded "train_data" folder to "data" folder
```
python code/lowlight_train.py 
```

## Contact
If you want to use our code, please contact Zhaoming Feng at 22303010013@stu.wit.edu.cn and cite our paper.


