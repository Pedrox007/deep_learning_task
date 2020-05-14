import cv2
import numpy as np
import os
import csv
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('saved_models/keras_test1_trained_model.h5')

project_dir = os.getcwd()
test_dir = os.path.join(project_dir, "dataset/test")
out_dir = os.path.join(project_dir, "output")
cor_dir = os.path.join(out_dir, "correct_images")

image_paths = None

print("Loading image paths...")
for root, __, files in os.walk(test_dir):
    image_paths = files
    break

data = []

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')

truth_file = open(os.path.join(out_dir, "test.preds.csv"), "w")
truth_writer = csv.DictWriter(truth_file, delimiter=',', fieldnames=["fn", "label"])
truth_writer.writeheader()

print("Start the prediction.")
for im_path in image_paths:
    save_path = os.path.join(cor_dir, im_path.split(".")[0] + ".png")

    im = cv2.imread(os.path.join("dataset/test", im_path))

    im_array = np.expand_dims(img_to_array(im), axis=0) / 255

    predict = model.predict_classes(im_array)

    orientation = encoder.classes_[predict]

    if orientation[0] == "rotated_left":
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif orientation[0] == "rotated_right":
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation[0] == "upside_down":
        im = cv2.rotate(im, cv2.ROTATE_180)

    data.append(im_array)
    truth_writer.writerow({"fn": im_path, "label": orientation[0]})

    cv2.imwrite(save_path, im)


truth_file.close()

print("Saving the image data on a numpy array.")
np.save(os.path.join(out_dir, "correct_images.npy"), data)