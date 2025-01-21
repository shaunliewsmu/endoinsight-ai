import tensorflow as tf
from tensorflow.keras import backend as K
from ghostNet import GhostNet
# from tools import *

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger

import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from BalancedDataGenerator import BalancedDataGenerator

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


grade0_aug_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range = 15,
    horizontal_flip = True,
    shear_range=0.2,
    zoom_range=0.2
)

x = np.load("./Datasets/02split_train_quality_imgs.npy")
y = np.load("./Datasets/02split_train_quality_labels.npy")
# x = np.load("./Datasets/03split_stratify_train_quality_imgs.npy")
# y = np.load("./Datasets/03split_stratify_train_quality_labels.npy")

train_datagen = BalancedDataGenerator(np.expand_dims(x,3), y, 
                                    grade0_aug_generator, 
                                    batch_size = BATCH_SIZE)
steps_per_epoch = train_datagen.steps_per_epoch

val_datagen = ImageDataGenerator()
x_test = np.load("./Datasets/02split_test_quality_imgs.npy")
y_test = np.load("./Datasets/02split_test_quality_labels.npy")
# x_test = np.load("./Datasets/03split_stratify_test_quality_imgs.npy")
# y_test = np.load("./Datasets/03split_stratify_test_quality_labels.npy")


# -----------------------------------Mdoel Training-------------------------------------- #
# Mdeol compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) # softmax
# model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['binary_accuracy']) # sigmoid
model.compile(optimizer= 'adam', loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False), metrics=['binary_accuracy']) # sigmoid

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

model.fit_generator(train_datagen, 
                    validation_data = val_datagen.flow(np.expand_dims(x_test,3), y_test, batch_size = BATCH_SIZE),
                    steps_per_epoch = steps_per_epoch,
                    validation_steps= len(x_test) // BATCH_SIZE, 
                    epochs= EPOCHS,
                    callbacks = callbacks_list,
                    verbose=1 )

import datetime
end = datetime.datetime.now()
print("========= End Time:", end, "===========")
print("========= Differenece: ", end - start, "===========")


# Save weights   
model.save_weights(f"./GhostNet/model/ghostNet_class_02_split_sigmoid_imbGene_BFL.hdf5")
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

