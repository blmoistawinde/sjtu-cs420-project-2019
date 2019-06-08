# VGG++ and VGGA
This directory contains the code and result for the VGG++ and VGGA of our report. Best accuracy is 73.08\%.

## Files
- preprocess_fer2013: preprocessing for fer2013
- fer.py data loading and preprocessing(grey to 3 channels)
- utils.py utility functions
- VGG++.py: training and testing for the VGG++ method
- VGG++_no_aug.py: training and testing for the VGG++(-data augmentation) method
- VGG++_no_ten.py: training and testing for the VGG++(-ten crop) method
- VGGA.py: training and testing for the VGGA(+FGSM) method
- Test_FER_adv.py: adversarial testing(trained model checkpoint required)
- /data: data storage
- /models: model definition
- /results: experiment results(original logs)
	- plot.py: draw plot according to logs
	- decay_acc_plot.jpg: epoch-test_accuracy plot(effect of weight decay)
	- lr_acc_plot.jpg: epoch-test_accuracy plot(effect of learning rate)

## Usage
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py
- python VGG++.py \[--bs 64\] \[--lr 0.01\]
- \[adversarial testing\] python Test_FER_adv.py (trained model checkpoint required)

Some code based on: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
