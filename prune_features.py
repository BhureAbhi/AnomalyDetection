from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributedDense ,LSTM,Reshape
from keras.regularizers import l2
from keras.optimizers import SGD,adam, Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import ConfigParser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
from datetime import datetime
from os.path import basename
import glob
import theano.sandbox
from theano.tensor import _shared

threshold = 0.0000001
feat_dim = 162

##################################################################################
# KERAS MODELS

print("Create Resize Model")
resize_model = Sequential()
resize_model.add(Dense(512, input_dim=feat_dim,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
resize_model.add(Dropout(0.1))
resize_model.add(Dense(128, init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#resize_model.add(Dropout(0.1))
resize_model.add(Dense(64,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#resize_model.add(Dropout(0.05))
resize_model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#resize_model.add(Dropout(0.1))

print("Create Score Model")
score_model = Sequential()
score_model.add(Dense(512, input_dim=feat_dim,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
score_model.add(Dropout(0.1))
score_model.add(Dense(128, init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#score_model.add(Dropout(0.1))
score_model.add(Dense(64,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#score_model.add(Dropout(0.05))
score_model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#score_model.add(Dropout(0.1))
score_model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))

####################################################################################
# FUNCTIONS

def load_weights(model, weight_path):                                   # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def listdir_nohidden(AllVideos_Path):                                   # To ignore hidden files
        # mention the extension of input files here.
        file_dir_extension = os.path.join(AllVideos_Path, '*.csv')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)


def generate_feature_vector(filename, inputPath, prunedOutputPath, scoresOutputPath, resize_model, score_model):
    input_file_path = inputPath + "/" + filename
    pruned_output_file_name = prunedOutputPath + "/" + filename[:-3] + 'txt'
    scores_output_file_name = scoresOutputPath + "/" + filename[:-3] + 'txt'
    csv_reader = csv.reader(open(input_file_path, "r"), delimiter=",")

    X = list(csv_reader)
    X = np.array(X).astype("float")

    Y = resize_model.predict(X)
    scores = score_model.predict(X)
    without_pruned_Y=[]
    pruned_Y = []
    pruned_scores = []
    for i in range(len(Y)):
        score = scores[i]
        without_pruned_Y.append(Y[i])
        if(score >= threshold):
            pruned_Y.append(Y[i])
            pruned_scores.append(score)
    #np.savetxt(pruned_output_file_name, pruned_Y, delimiter=' ', fmt='%f')
    #np.savetxt(scores_output_file_name, pruned_scores)
    np.savetxt(pruned_output_file_name, without_pruned_Y, delimiter=' ', fmt='%f')
    np.savetxt(scores_output_file_name, scores)
    return


weight_path = 'NEW_UCSD_PED1_output/weightsAnomalyL1L2_40000.mat'       # path to weights
resize_model = load_weights(resize_model, weight_path)
score_model = load_weights(score_model, weight_path)

AllClassPath='NEW_UCSD_PED1_STIP_106_8/'                                  # path to videos' stip folder
All_class_files= listdir(AllClassPath)
All_class_files.sort()

AbnormalPath = os.path.join(AllClassPath, All_class_files[0])
NormalPath = os.path.join(AllClassPath, All_class_files[1])
PrunedAbnormalOutputPath = 'new_pruned_features_8/abnormal/'
PrunedNormalOutputPath = 'new_pruned_features_8/normal/'
ScoresAbnormalOutputPath = 'new_anomaly_scores_8/abnormal/'
ScoresNormalOutputPath = 'new_anomaly_scores_8/normal/'

Abnormal_Videos=sorted(listdir_nohidden(AbnormalPath))
Abnormal_Videos.sort()
Normal_Videos=sorted(listdir_nohidden(NormalPath))
Normal_Videos.sort()

for video in Abnormal_Videos:
    generate_feature_vector(video, AbnormalPath, PrunedAbnormalOutputPath, ScoresAbnormalOutputPath, resize_model, score_model)
for video in Normal_Videos:
    generate_feature_vector(video, NormalPath, PrunedNormalOutputPath, ScoresNormalOutputPath, resize_model, score_model)
