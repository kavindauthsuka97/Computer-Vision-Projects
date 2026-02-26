import os                      # lets us work with folders/files (list files, build paths)
import pickle                  # used to save/load Python objects (we save the trained model)

from skimage.io import imread  # reads an image from disk into an array
from skimage.transform import resize  # resizes images to a fixed size
import numpy as np             # numerical arrays + math utilities
from sklearn.model_selection import train_test_split  # splits data into train/test sets
from sklearn.model_selection import GridSearchCV      # tries many hyperparameters automatically
from sklearn.svm import SVC    # Support Vector Classifier (SVM model)
from sklearn.metrics import accuracy_score  # computes accuracy of predictions


# prepare data
input_dir = '/home/phillip/Desktop/todays_tutorial/19_parking_car_counter/code/clf-data'  # dataset root folder
categories = ['empty', 'not_empty']  # folder names = class names

data = []    # will store the flattened image features
labels = []  # will store the class label (0 or 1) for each image

for category_idx, category in enumerate(categories):  # loop over classes; category_idx becomes 0 then 1
    for file in os.listdir(os.path.join(input_dir, category)):  # loop over every image file in that class folder
        img_path = os.path.join(input_dir, category, file)  # full path to the image file
        img = imread(img_path)                              # read image as an array (H x W x C or H x W)
        img = resize(img, (15, 15))                         # resize to 15x15 so every sample has same size
        data.append(img.flatten())                          # flatten 15x15(×channels) into 1D feature vector
        labels.append(category_idx)                         # store the label (0 for empty, 1 for not_empty)

data = np.asarray(data)      # convert list of vectors into a NumPy array (N x features)
labels = np.asarray(labels)  # convert labels list into a NumPy array (N,)


# train / test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels,            # features and labels to split
    test_size=0.2,           # 20% goes to test set
    shuffle=True,            # shuffle before splitting
    stratify=labels          # keep class balance similar in train and test
)


# train classifier
classifier = SVC()  # create an SVM classifier with default settings

parameters = [
    {                          # hyperparameter grid to try
        'gamma': [0.01, 0.001, 0.0001],  # how flexible the decision boundary is (RBF kernel uses gamma)
        'C': [1, 10, 100, 1000]          # regularization strength (bigger C = fits training data more)
    }
]

grid_search = GridSearchCV(
    classifier,               # model to tune
    parameters                # values to try
)                             # (defaults: cv=5 folds, scoring=estimator default)

grid_search.fit(x_train, y_train)  # train multiple models across parameter combinations using CV

# test performance
best_estimator = grid_search.best_estimator_  # pick the best-performing model from grid search

y_prediction = best_estimator.predict(x_test)  # predict labels for the test set

score = accuracy_score(y_prediction, y_test)   # compute accuracy (how many predictions were correct)

print('{}% of samples were correctly classified'.format(str(score * 100)))  # print accuracy percentage

pickle.dump(best_estimator, open('./model.p', 'wb'))  # save the trained best model to model.p (binary file)