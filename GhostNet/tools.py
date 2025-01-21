from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from GhostNet.ghostNet import GhostNet
from keras.models import load_model

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve, f1_score
from tensorflow.keras.models import load_model


'''
Function for plot ROC_AUC curve
'''

def plot_roc_auc(test_labels, test_predict):
  fpr, tpr, threshold = roc_curve(test_labels, test_predict, pos_label = 1)
  roc_auc = auc(fpr, tpr)
  plt.stackplot(fpr, tpr, color = "steelblue", alpha = 0.3, edgecolor = "black")
  plt.plot(fpr, tpr, lw = 1, label = "ROC(area = %0.2f)" % (roc_auc))

  plt.plot([0,1],[0,1], color = "r", linestyle = "--", alpha = 0.5)
  plt.text(0.5,0.4,"ROC curve (area = %0.2f) "%( roc_auc))
  plt.xlabel("FPR (False Positive Rate)")
  plt.ylabel("TPR (True Positive Rate)")
  plt.title("Receiver Operating Charactistic, ROC(AUC=%0.2f) "% (roc_auc))
  plt.show()



'''

Using "softmax" activation on the top of model, thus don't need threshold for classifying labels.

test_model_generator: while use "DataGenerator" to load data in model training, use this function to test the model performance
test_model_from_npy: while use ".npy" file to load data, use this function to test the model performance 

'''

def test_model_generator(path, CLASSES, test_labels, validation_generator):
  model = GhostNet((256, 256, 1), CLASSES).build()
  model.load_weights(path)


  import datetime
  start = datetime.datetime.now()
  print("====== Start Time:", start, "========")
  test_predict = model.predict_generator(validation_generator)
  end = datetime.datetime.now()
  print("======= End Time:", end, "========")
  print("======= Differenece: ", end - start, "========" )
  test_predict = test_predict.argmax(axis=-1)


  precisions, recalls, thresholds = precision_recall_curve(test_labels,test_predict)
  fpr, tpr, threshold = roc_curve(test_labels, test_predict, pos_label = 1)


  print("Acc:      ", accuracy_score(test_labels,test_predict))
  print("AUC_ROC:  ", auc(fpr, tpr))
  print("AUC_PR:   ", auc(recalls, precisions))
  print("f1_score: ", f1_score(test_labels, test_predict, average='weighted'))

  plot_roc_auc(test_labels, test_predict)
  print(confusion_matrix(test_labels,test_predict))
  classify_report = classification_report(test_labels, test_predict, target_names=["Grade 0","Grade 1"])
  print("classification report: \n", classify_report) 


def test_model_from_npy(path,CLASSES, test_labels, test_imgs):
  model = GhostNet((256, 256, 1), CLASSES).build()
  model.load_weights(path)

  test_predict = model.predict(np.expand_dims(test_imgs,3))
  test_predict = test_predict.argmax(axis=-1)


  precisions, recalls, thresholds = precision_recall_curve(test_labels,test_predict)
  fpr, tpr, threshold = roc_curve(test_labels, test_predict, pos_label = 1)


  print("Acc:      ", accuracy_score(test_labels,test_predict))
  print("AUC_ROC:  ", auc(fpr, tpr))
  print("AUC_PR:   ", auc(recalls, precisions))
  print("f1_score: ", f1_score(test_labels, test_predict, average='weighted'))

  plot_roc_auc(test_labels, test_predict)
  print(confusion_matrix(test_labels,test_predict))
  classify_report = classification_report(test_labels, test_predict, target_names=["Grade 1","Grade 0"])
  print("classification report: \n", classify_report) 


  
'''

Using "sigmoid" activation on the top of model, thus need threshold for classifying labels.

'''

def test_model_report_threshold(train_labels, test_labels, train_predict, test_predict):
  precisions, recalls, thresholds = precision_recall_curve(train_labels,train_predict)
  f1_scores = (2 * precisions * recalls) / (precisions + recalls)
  best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
  best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
  best_threshold = thresholds[best_f1_score_index]
  print("best threshold:             ", best_threshold)

  precisions, recalls, thresholds = precision_recall_curve(test_labels,test_predict)
  fpr, tpr, threshold = roc_curve(test_labels, test_predict, pos_label = 1)


  print("Acc based on Train best threshold:", accuracy_score(test_labels,np.where(test_predict >= best_threshold,1,0)))
  print("AUC_ROC:                          ", auc(fpr, tpr))
  print("AUC_PR:                           ", auc(recalls, precisions))
  print("f1_score:                         ", f1_score(test_labels, np.where(test_predict >= best_threshold,1,0), average='weighted')) 

  plot_roc_auc(test_labels, test_predict)
  print(confusion_matrix(test_labels,np.where(test_predict >=best_threshold,1,0)))
  classify_report = classification_report(test_labels, np.where(test_predict >=best_threshold,1,0), target_names=["Grade 1","Grade 0"])
  print("classification report: \n", classify_report)