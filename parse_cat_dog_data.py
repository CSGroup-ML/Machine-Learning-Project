import numpy as np
import cv2

from random import randint

class cats_and_dogs(object):

    cat_pictures_train = []
    dog_pictures_train = []
    cat_pictures_test = []
    dog_pictures_test = []
    train_examples = 0
    test_examples = 0
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    def __init__(self, train_amount, test_amount):
        self.train_examples = train_amount
        self.test_examples = test_amount

        for i in range(0, int(train_amount / 2)):
            image = "training_images/train_new/cats/cat." + str(i) + ".jpg"
            picture = cv2.imread(image, 1)
            picture = cv2.resize(picture, (32, 32))
            #picture = np.reshape(picture, (32 * 32 * 3))
            self.cat_pictures_train.append(picture)
        print("Cat Training Pictures Formatted")

        for i in range(0, int(train_amount / 2)):
            image = "training_images/train_new/dogs/dog." + str(i) + ".jpg"
            picture = cv2.imread(image, 1)
            picture = cv2.resize(picture, (32, 32))
            #picture = np.reshape(picture, (32 * 32 * 3))
            self.dog_pictures_train.append(picture)
        print("Dog Training Pictures Formatted")

        for i in range(11500, int(test_amount / 2) + 11500):
            image = "training_images/test_new/cats/cat." + str(i) + ".jpg"
            picture = cv2.imread(image, 1)
            picture = cv2.resize(picture, (32, 32))
            #picture = np.reshape(picture, (32 * 32 * 3))
            self.cat_pictures_test.append(picture)
        print("Cat Test Pictures Formatted")

        for i in range(10000, int(test_amount / 2) + 10000):
            image = "training_images/test_new/dogs/dog." + str(i) + ".jpg"
            picture = cv2.imread(image, 1)
            picture = cv2.resize(picture, (32, 32))
            #picture = np.reshape(picture, (32 * 32 * 3))
            self.dog_pictures_test.append(picture)
        print("Dog Test Pictures Formatter")

        self.make_training_and_test_sets()

    def make_training_and_test_sets(self):
        cat_index = 0
        dog_index = 0
        for i in range(0, self.train_examples):
            x = randint(0, 1)
            if ((x == 0 or cat_index == int(self.train_examples/2)) and dog_index < int(self.train_examples / 2)):
                self.X_train.append(self.dog_pictures_train[dog_index])
                self.Y_train.append([0])
                dog_index += 1
                print("Dog")
            else:
                self.X_train.append(self.cat_pictures_train[cat_index])
                self.Y_train.append([1])
                cat_index += 1
                print("Cat")

        cat_index = 0
        dog_index = 0
        for i in range(0, self.test_examples):
            x = randint(0, 1)
            if ((x == 0 or cat_index == int(self.test_examples/2)) and dog_index < int(self.test_examples / 2)):
                self.X_test.append(self.dog_pictures_test[dog_index])
                self.Y_test.append([0])
                dog_index += 1
            else:
                self.X_test.append(self.cat_pictures_test[cat_index])
                self.Y_test.append([1])
                cat_index += 1


    def get_data(self):
        return (self.X_train, self.Y_train, self.X_test, self.Y_test)

    def get_data_set_for_predictions(self):
        return (self.X_train, self.Y_train)
