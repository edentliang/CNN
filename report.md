---
typora-copy-images-to: ./acc.png
---



## Abstract

Using VGG16 and ResNet18 to make image prediction. 



## Process

### Raw Data Process

Using Data augmentation to make regularitzation

#### train dataset

* randomly crops the 32x32
* randomly horizontonal flips

####test dataset

* resize to 32x32
* crops the center with the size 32x32



### Set Hyperparamater 

* batch_size: 256
* epochs: 300
* loss function: cross entropy
* optimizer: SGD 
  * weight_decay 0.0001
  * momentum: 0.9
  * initial learning rate: 0.1
    - learning rate shrink 0.1 when epoch reach 90th, 175th and 225 respectively



###Network Preparation

####VGG16

There are 16 layers, for each layers has *[64, 64, M, 128, 128, M, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M]* channels with 3x3 filters, where 'M' is the maxpool.

For the classifier, there are 3 fully connected layer *[4096, 4096, 10]*. 

Using ReLU and dropout after every FC



#### ResNet18

There are 18 layers in ResNet18. Specificially,  there are 8 blocks which have *[64, 64, 128, 128, 256, 256, 512, 512]* channels, and each block has two layers with szie 1x1 and 3x3 respectively.

And there are only one fuuly connected layer at the end with the size 10



##Result

#### VGG16

![vgg18_test_acc](/Users/haoxingliang/Desktop/cv/code/result/report/report/vgg18_test_acc.png)

#### ResNet18

![acc](/Users/haoxingliang/Desktop/cv/code/result/report/report/acc.png)

