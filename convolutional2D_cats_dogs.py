import keras

from keras.preprocessing.image import  ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout


img_width, img_height = 150, 150
training_data = 'training_images/train' #String representation of the directory to search for training data
testing_data = 'training_images/test'   #String representation of the directory to search for testing data
training_samples = 20000                #Number of training samples in the current experiment
test_samples = 5000                     #Number of testing samples in the current experiment
epoch = 50   #Number of epochs: full instances of processing all images once (forward passes + backward passes)
batch_size = 16 #Number of examples to process per batch

generate_image_data = ImageDataGenerator(rescale=1./255) #Instantiate ImageDataGenerator class which is used to format images according
                                           #to the following member function: flow_from_directory

#flow_from_directory() in this case takes several arguments:
# 1.)Directory that contains subdirectories each representing the classes in the current classification problem
# 2.) target_size or size to adjust images read from the directories to
# 3.) color_mode set to read color images with the string 'rbg'
# 4.) classes: specifies the labels for the pertaining classes to identify based on the names of the subdirectories
# 5.) class_mode specifies the type of classification scheme used in this case, binary for identification of two classes
# 6.) batch_size: the number of images per batch in iterations of processing
training = generate_image_data.flow_from_directory(training_data, target_size=(img_width, img_height), color_mode='rgb',
                                                   classes=['cats', 'dogs'], class_mode='binary', batch_size=32)
testing = generate_image_data.flow_from_directory(testing_data, target_size=(img_width, img_height), color_mode='rgb',
                                                  classes=['cats', 'dogs'], class_mode='binary', batch_size=32)

model = Sequential() #Sequential model (Neural Network with layers arranged in sequence) The architecture of this network
                     #is defined in the following statements as a network with three convolutional layers followed by
                     #flattening of the results of convolution and two layers yielding a binary output.

#Add a 2-Dimensional Convolutional layer to the model that performs spatial convolution over images
#32 filters (kernels) of size 3 x 3 applied to the input layer (as first input: images vectorized)
#input_shape refers to the dimensions of the input and expects in this case input of img_width x img_height x 3
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu')) #Activation function applied to this layer is Rectified Linear Unit
model.add(MaxPooling2D(pool_size=(2, 2))) # MaxPooling for two dimensions: greatest value from every 2 x 2 resulting square
                                         # from convolution will be propogated and the rest discarded.

#Add a second layer to the CNN with the same number of filters and their size, activation function and pooling method
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Third layer in the CNN with twice the number of filters applied to the input for this layer.
model.add(Conv2D(64, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Concatenate all output layers by columns into a single vector
model.add(Dense(64)) # Add a layer with 64 neurons
model.add(Activation('relu')) # Rectified Linear Unit function for activation of the dense layer
model.add(Dropout(0.5)) #Dropout used to prevent overfitting with a rate of 0.5 or half of the input values set to 0
model.add(Dense(1)) # One more layer with a single neuron in this binary classification network
model.add(Activation('sigmoid')) #Use the sigmoid function as the activation function in this neuron. Appropriate for
                                 #the distinction between two different classes given sigmoid(theta^T*X) = 1 / (1 + e^(theta^T*X))
                                 #and this gives values in the range of 0 < sigmoid < 1.

#Configure learning system for the network above specifying the loss function as of binary cross entropy, an optimizer
#based on rmsprop and to return information about the accuracy of the network.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(generator=training, steps_per_epoch=200, epochs=50, verbose=2, validation_data=testing, validation_steps=800)
