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

# ---------------------Build Model------------------------- #

# config
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# sess = tf.Session(config=config)
# tf.keras.backend.set_session(sess)

# hyper-parameter
BATCH_SIZE = 24
EPOCHS = 20
CLASSES = 1       

# buil model
model = GhostNet((256,256,1),CLASSES).build(True)
# model.summary()

# -----------------Load Data------------------ #
# Train
train_labels = np.load("../Datasets/all_labels_samples_train_full.npy")
print(len(train_labels))
train_imgs = np.load("../Datasets/all_imgs_samples_train_full.npy")
# from tensorflow.keras.utils import to_categorical
# train_labels = to_categorical(train_labels)

print("Trianing Dataset Done!")

# Test
test_labels = np.load("../Datasets/all_labels_samples_test_full.npy")
test_imgs = np.load("../Datasets/all_imgs_samples_test_full.npy")
# from tensorflow.keras.utils import to_categorical
# test_labels = to_categorical(test_labels)

print("Validation Dataset Done!")


# -----------------------------------Model Training-------------------------------------- #
# model compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['binary_accuracy'])


model_name = "./model/GhostNet_0923"
filepath        = model_name + ".hdf5"
model_checkpoint  = ModelCheckpoint(filepath, 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=True, 
                                mode='min')

# tensorboard = TensorBoard(log_dir='./tmp/log/{}'.format(model_name),
#                           histogram_freq=0, write_graph=True, write_images=True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                                  patience = 1)

earlystop       = EarlyStopping(monitor='val_loss', patience=3)

# csv_logger      = CSVLogger(model_name +'.csv')

callbacks_list  = [ reduce_lr, earlystop, model_checkpoint]
# train
print("Begin training !")


import datetime
start = datetime.datetime.now()
print("====== Start Time:", start, "========")

result = model.fit(np.expand_dims(train_imgs,3),train_labels,
                    # validation_split = 0.2,
	                  validation_data= (np.expand_dims(test_imgs,3),test_labels), 
                    batch_size = BATCH_SIZE,
	                  epochs = EPOCHS,
                    callbacks = callbacks_list,
                    # validation_steps=len(test_imgs) // BATCH_SIZE,
                    verbose=1)

end = datetime.datetime.now()
print("======= End Time:", end, "========")
print("======= Differenece: ", end - start, "========" )


# save weights           
model.save_weights(f"./model/ghostNet_0923.hdf5")

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



# -----------------------------------Plot Training------------------------------- #
# plt.figure()
# plt.plot(result.epoch, result.history['acc'], label="acc")
# plt.plot(result.epoch, result.history['val_acc'], label="acc")
# plt.scatter(result.epoch, result.history['acc'], marker='*')
# plt.scatter(result.epoch, result.history['val_acc'])
# plt.legend(loc='under right')
# plt.savefig("./model/acc_final_dataAug.png")
# # plt.show()

# plt.figure()
# plt.plot(result.epoch, result.history['loss'], label="loss")
# plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
# plt.scatter(result.epoch, result.history['loss'], marker='*')
# plt.scatter(result.epoch, result.history['val_loss'])
# plt.legend(loc='upper right')
# plt.savefig("./model/loss_final_dataAug.png")
# # plt.show()

