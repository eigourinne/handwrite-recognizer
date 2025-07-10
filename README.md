# a simple handwrite-number recognizer&classifier

## the project mainly consists of a recognizer and a classifier:

the classifier based on res-cnn (channel attention + spiral attention) network, trained by enhanced-mnist handwrite-numbers databases(randomly rotate or erase or add guass blur)

the recognizer use opencv's findcontour method to find the place of numbers in the picture, then use rotate rectangle to detect the zone, reshape it to 28*28 pixels and pass it to the classifier

by the way, divide.py and rotate.py are used for manly adjust origin picture, because the recognizer based on traditional machine learning (ML) method, can't solve too complex picture

## requirements:

- pytorch(if necessary, nvidia gpu is the best choice for ai-related learner)
- python-pytorch-cuda(my programming environment can use nvidia to accelerate training, if not, you can use google's colab to set up the project)
- nccl(if you have a package manager, it 'll be automatively install while installing cuda or torch)
- torchvision(another dependence for this project)
- PIL(used for convert rgb 3D picture to gray picture)
- miniconda3/...(simple virtual environment, but I recommend python-virtual, 'cause conda is too large)
- cuda(no one would doubt this power for ai-related programming, it can greatly improve training speed)
- opencv(another strong package for cv, many useful functions could be provided)
- python-opencv(opencv package for python)
- etc...(I don't wanna talk anymore, if you only have integrated gpu, use colab instead,'cause torch-cpu's training cost too much times)

## how to use:

1. run train.py to train the classifier, it may take hours to finish

2. run main.py, after that you need to input the path of picture on the console under the same directory, then it will use recognizer and classier to detect the picture and save the result

3. my project just use simple mnist databases to train, so badly act on complex picture, maybe yolo or other end-to-end systems behave much better, but this peoject is just my first machine learning's "Hello World", I'm feel sorry for the poor accuracy 

  ### examples:

  - input picture:
  - ![image](https://github.com/user-attachments/assets/0831cc29-4804-42d3-9d35-075429404d9f)
  - output picture:
  - ![image](https://github.com/user-attachments/assets/0e80acfc-ef91-4bf6-85ec-804dcc3305eb)
