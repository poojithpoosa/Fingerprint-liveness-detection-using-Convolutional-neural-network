### Fingerprint-liveness-detection-using-Convolutional-neural-network


According to recent studies, fingerprint scanners constructed of plastic, clay, Play-Doh, silicone, or gelatine may be fooled using a few easy procedures. Liveness detection techniques examine the liveliness of fingers being recorded for registration or authentication, thereby protecting against spoofing. The ridge-valley structure of fingerprint pictures is used to analyse the texture of the fingerprints. A fake finger has a different texture when put on a fingerprint scanner because of the features of the material. With the help of the Gabor filter and Weber local descriptor, I have captured those hidden textures and features in the fingerprints by using the SOCOFing dataset as input. Using the textured data, I have classified the dataset using the CNN model. I have got 97.34% accuracy and lost 0.127.

link of the dataset: https://www.kaggle.com/datasets/ruizgara/socofing


### Methodology 
In this project, I have developed a fingerprint liveness detection system using convolutional neural networks. I have used a SOCOFing dataset which is a standard dataset for checking the liveness of the fingerprints. This project is based on computer vision technology where we make computer understand the content of the digital image. In this project I have used 2 preprocessing techniques known as Gabor filter and web local descriptor. Many CV algorithms depend on local features and related descriptors, which are compact vector representations of a small area, to improve the image's hidden textures. These methods are better equipped to deal with changes in size, rotation, and occlusion because to the use of local features. For feature extraction I have used 3-layer CNN layer and for classification I have used 2-layer dense layers.

## project work flow
![image](https://user-images.githubusercontent.com/61981756/199001340-4664cbcc-8d47-4544-8499-7faaf40f453a.png)

## technologies :

* python
* tensorflow
* opencv
* scipy
* matplotlib

results:

![image](https://user-images.githubusercontent.com/61981756/199001409-e6ce6b8e-ecfe-4e87-83db-89a218ceae39.png)

![image](https://user-images.githubusercontent.com/61981756/199001439-7dca0d9d-9ab6-4d98-bf2f-8f94b777aee6.png)
