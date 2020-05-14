# Deep Learning Task

Required libs:
* Keras
* Numpy
* SKLearn
* OpenCV2

Steps to run:
* Create a folder called `dataset` and put the test and train data inside of it
* To create and train a model, open the `build_model.py` and put the model name on the line `21` and the number of epochs on the line `17`. After this you just run the file that after the training the model will be saved on the `output` directory.
* To make predictions with the model created, go to the `predict.py` and edit the line `9` putting your model name. After this, just run this file that the correct images will be saved on the  `output/correct_images` and two files will be created:
  * test.preds.csv (This file has the labels predicted associated with the image file name)
  * correct_images_np_array.zip (This file has the numpy arrays from each correct image)

Notes:
* My files is already inside on the output directory:
  * `output/correct_images.zip`
  * `output/correct_images_np_array.zip`
  * `test.preds.csv`
* My approach was first I trained the model on the cifar10 for 20 epochs to see how the `build_model.py` file works. After this I tried to figure out how I could access the dataset data to make this work when passed to the model. When I figured out I passed the code to the `build_model.py` and started a new training with only  epochs just to test. The training was good and I reached 90+% of accuracy without change the optimizer or the initial learning rate. After this I started to think how I could make the predictions using the trained model. After make some changes on the `build_model.py`again to it save the encoded labels, I could use this file to decode the predict label and after this I just create the `predict.py` to put the predict code inside of it.
