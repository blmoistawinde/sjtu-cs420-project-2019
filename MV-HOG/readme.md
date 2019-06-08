# MV-HOG

1. Download Fer2013 dataset and the Face Landmarks model
	- [Kaggle Fer2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
    - [Dlib Shape Predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. Move the files `fer2013.csv` and `shape_predictor_68_face_landmarks.dat` in the root folder of this package.

3. `python convert_fer2013_to_images_and_landmarks.py`

4. Launch training: `python train.py --train=yes`

5. Train and evaluate: `python train.py --train=yes --evaluate=yes`

Some code based on: https://github.com/amineHorseman/facial-expression-recognition-using-cnn