import tensorflow as tf
from tensorflow.keras import backend as K
from ghostNet import GhostNet
from tools import *

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger

import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


print(f"Tensorflow version: {tf.__version__}")

# ---------------------Model------------------------- #

# config
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Hyper-parameter
BATCH_SIZE = 24
EPOCHS = 20
# CLASSES = 2      # softmax
CLASSES = 1        # sigmoid

# Model
model = GhostNet((256,256,1),CLASSES).build(True)
model.summary()

# ----------------- Data ------------------ #
# # data augmentation
# train_datagen = ImageDataGenerator(rotation_range=10,horizontal_flip=True)
# print("aug!!!")
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Training dataset
train_generator = train_datagen.flow_from_directory("../Datasets/Duke_Images_train_test_full/train", target_size=(256, 256),
                                                    color_mode = "grayscale",
                                                    shuffle=True,
                                                    class_mode = 'binary', # sigmoid
                                                    # classes=None,  # softmax
                                                    batch_size=BATCH_SIZE)
print("Trianing Dataset Done!")

# Validation dataset
validation_generator = val_datagen.flow_from_directory("../Datasets/Duke_Images_train_test_full/test", target_size=(256, 256),
                                                       color_mode = "grayscale",
                                                       shuffle=True,
                                                       class_mode = 'binary', # sigmoid
                                                      #  classes=None, # softmax
                                                       batch_size=BATCH_SIZE)
print("Validation Dataset Done!")

from sklearn.utils import class_weight
print("class weight")
class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_generator.classes),
                                        y = train_generator.classes                                                    
                                    )
class_weights = dict(zip(np.unique(train_generator.classes), class_weights))

# class_weights = {0: 100, 1: 1}



# -----------------------------------Mdoel Training-------------------------------------- #
# Mdeol compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) # softmax
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['binary_accuracy']) # sigmoid

# Model checkpoint
# modelname = './model/GhostNet_class_0925'
# filepath        = modelname + ".hdf5"
# checkpoint      = ModelCheckpoint(filepath, 
#                                   monitor='val_loss', 
#                                   verbose=0, 
#                                   save_best_only=True, 
#                                   mode='min')


reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                                  patience = 1)

earlystop       = EarlyStopping(monitor='val_loss', patience=3)

# csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [reduce_lr, earlystop]


# Train
print("Begin training !")

import datetime
start = datetime.datetime.now()
print("========= Start Time:", start, "===========")

result = model.fit_generator(train_generator,
	                  validation_data= validation_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
	                  epochs= EPOCHS,
                    callbacks = callbacks_list,
                    class_weight = class_weights,
                    validation_steps=validation_generator.samples // BATCH_SIZE,
                    verbose=1)

import datetime
end = datetime.datetime.now()
print("========= End Time:", end, "===========")
print("========= Differenece: ", end - start, "===========")


print("steps_per_epoch = ", train_generator.samples // BATCH_SIZE)
print("validation_steps = ", validation_generator.samples // BATCH_SIZE)


# Save weights   
model.save_weights(f"./model/ghostNet_class_02split_sigmoid_classweight_v2.hdf5")
# model.save_weights(f"./model/ghostNet_class_0925_softmax.hdf5")

# ----------------------------------- Accuracy ------------------------------- #
# train_predict = model.predict(np.expand_dims(train_imgs,3))
# test_predict = model.predict(np.expand_dims(test_imgs,3))
# test_model_report(train_labels, test_labels, train_predict, test_predict)
# train_predict = model.predict_generator(train_generator)
# train_predict = train_predict.argmax(axis=-1)

# test_predict = model.predict_generator(validation_generator)
# test_predict = test_predict.argmax(axis=-1)

# np.save("./test_Flow_predict_kaggle_final.npy", np.array(test_predict))
# np.save("./train_Flow_predict_kaggle_final.npy", np.array(train_predict))
# plot_roc_auc(test_labels, test_predict)



# -----------------------------------Visulization Training Curve------------------------------- #
# plt.figure()
# plt.plot(result.epoch, result.history['acc'], label="acc")
# plt.plot(result.epoch, result.history['val_acc'], label="acc")
# plt.scatter(result.epoch, result.history['acc'], marker='*')
# plt.scatter(result.epoch, result.history['val_acc'])
# plt.legend(loc='under right')
# plt.savefig("./model/acc_class_weight2.png")
# # plt.show()

# plt.figure()
# plt.plot(result.epoch, result.history['loss'], label="loss")
# plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
# plt.scatter(result.epoch, result.history['loss'], marker='*')
# plt.scatter(result.epoch, result.history['val_loss'])
# plt.legend(loc='upper right')
# plt.savefig("./model/loss_class_weight2.png")
# plt.show()

