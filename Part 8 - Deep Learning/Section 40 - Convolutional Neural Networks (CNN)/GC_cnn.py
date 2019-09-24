# Convolutional Neural Network
# Data is ready - no Preprocessing needed

# Build the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential # Initialize NN
from keras.layers import Conv2D # Convolution step
from keras.layers import MaxPooling2D # Pooling step
from keras.layers import Flatten # Get Data Flat for ANN
from keras.layers import Dense # Add fully connected layers into an ANN

# Initialize CNN
classifier = Sequential()

# Add Convolutional Layer
# 32 Feature Detectors
# 3 rows x 3 cols size of detector
# input_shape: uses tensorflow format for images 64x64 dimension of images
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# Add Pooling Layer
# 2,2 recommended to not lose ientified features
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Add other convolutional Layer for performance results on accuracy
classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Add Flat Layer
classifier.add(Flatten())

# Full Connection - Connect to an ANN
# Add Hidden Layer
# 128 experimentation
classifier.add(Dense(units = 128, activation = 'relu'))
# Second Hidden Layer
classifier.add(Dense(units = 64, activation = 'relu'))
# Add Output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit CNN
# Avoid Overfitting 
# Image augmentation - Reduce Overfitting -Enrich Trainig Set
# Code directly from keras documentation - image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")

# Save the model 
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
classifier.save("DogCatClassifier.hdf5", overwrite=True)


