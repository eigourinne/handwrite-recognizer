handwrite-number-recognizer

#
mainly consists of a recognizer and a classifier:

the classifier based on res-cnn (channel attention + spiral attention) network, trained by enhanced-mnist handwrite-numbers databases

the recognizer use opencv's findcontour method to find the place of numbers in the picture, then use rotate rectangle to detect the zoom, shape it to 28*28 pixels and pass it to the classifier

by the way, divide.py and rotate.py are used for manly adjust origin picture, because the recognizer based on traditional machine learning (ML) method, can't solve too complex picture
#

#
requirements:
- pytorch
- torchvision
- PIL
- conda/...(virtual environment)
- cuda
- python-opencv
- etc...
#

#
how to use:
1. run train.py to train the classifier
2. run main.py, after that you need to input the path of picture, then it will use recognizer and classier to detect the picture and save the result
#

#
example:
input picture:

![image](https://github.com/user-attachments/assets/0831cc29-4804-42d3-9d35-075429404d9f)

output picture:

![image](https://github.com/user-attachments/assets/0e80acfc-ef91-4bf6-85ec-804dcc3305eb)

#
