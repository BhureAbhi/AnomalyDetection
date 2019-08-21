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

batchsize = 52
nfeats = 106
feat_dim = 162
Num_Normal = 26
Num_abnormal = 28
num_iters = 20000

#########################################################################################################
# KERAS MODEL

print("Create Model")
model = Sequential()
model.add(Dense(512, input_dim=feat_dim,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(64,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#model.add(Dropout(0.05))
model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))

############################################################################################################
# FUNCTIONS

def load_model(json_path):                                                      # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):                                           # Function to load the model weights
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

def save_model(model, json_path, weight_path):                                  # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

def custom_objective(y_true, y_pred):
    print('Custom Objective function')

    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)
    n_seg = nfeats                                                              # Because we have nfeats segments per video.
    nvid = batchsize
    n_exp = nvid / 2
    Num_d=n_seg*nvid


    sub_max = T.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = T.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1=T.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = T.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for ii in xrange(0, nvid, 1):
        # For Labels
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])  # Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[nfeats-1:]
        z_44 = z3[:nfeats+1]
        z = z_22 - z_44
        z = z[1:nfeats]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    # sub_max[Num_d:] means include all elements after Num_d.
    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer nfeats for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:n_exp]


    indx_nor = theano.tensor.eq(F_labels, nfeats).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of nfeats segments of normal videos F_labels=nfeats for normal video
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor=n_exp

    Sub_Nor = sub_score[indx_nor] # Maximum Score for each of normal video
    Sub_Abn = sub_score[indx_abn] # Maximum Score for each of abnormal video

    z = T.ones_like(y_true)
    for ii in xrange(0, n_Nor, 1):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = T.mean(z, axis=-1) +  0.00008*T.sum(sub_sum_l1) + 0.00008*T.sum(sub_l2)  # Final Loss f

    return z


######## Load Training Dataset ###########
def load_dataset_Train_batch(AbnormalPath, NormalPath):
    # print("Loading training batch")
    total_nfeats = 0                # total number of features for the entire batch
    n_exp = batchsize/2             # Number of abnormal and normal videos

    # We assume the features of abnormal videos and normal videos are located in two different folders.
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal-n_exp:]  # Indexes for randomly selected Abnormal Videos
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal-n_exp:]      # Indexes for randomly selected Normal Videos

    # AllFeatures is the matrix containing features of abnormal and normal videos stacked together vertically.
    # Video_count keeps count of number of video files processed.
    AllFeatures = []
    Video_count=-1

    #################################################################################
    # print("Loading Abnormal videos Features...")

    AllVideos_Path = AbnormalPath
    def listdir_nohidden(AllVideos_Path):                   # To ignore hidden files
        # mention the extension of input files here.
        file_dir_extension = os.path.join(AllVideos_Path, '*.csv')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos=sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()

    for iv in Abnor_list_iter:
        Video_count=Video_count+1
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])

        csv_file = open(VideoPath, "r")

        csv_reader = csv.reader(csv_file, delimiter=',')
        count = -1;
        VideoFeatures = []

        for row in csv_reader:
            feat_row1 = np.float32(row)
            count = count + 1
            if(count==0):
                VideoFeatures = feat_row1
            if(count > 0):
                VideoFeatures = np.vstack((VideoFeatures, feat_row1))

        csv_file.close()

        if Video_count == 0:
            AllFeatures = VideoFeatures
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatures))

    # print(" Abnormal Features  loaded")
    ################################################################################

    ################################################################################
    # print("Loading Normal videos...")

    AllVideos_Path =  NormalPath
    def listdir_nohidden(AllVideos_Path):                   # To ignore hidden files
        # mention the extension of input files here.
        file_dir_extension = os.path.join(AllVideos_Path, '*.csv')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos = sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()

    for iv in Norm_list_iter:
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        csv_file = open(VideoPath, "r")

        count = -1
        VideoFeatures = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            feat_row1 = np.float32(row)
            count = count + 1
            if(count==0):
                VideoFeatures = feat_row1
            if(count > 0):
                VideoFeatures = np.vstack((VideoFeatures, feat_row1))

        AllFeatures = np.vstack((AllFeatures, VideoFeatures))
    # print(" Normal Features  loaded")
    ###############################################################################

    AllLabels = np.zeros(nfeats*batchsize, dtype='uint8')
    th_loop1=n_exp*nfeats
    th_loop2=n_exp*nfeats-1

    for iv in xrange(0, nfeats*batchsize):
            if iv< th_loop1:
                AllLabels[iv] = int(0)  # All instances of abnormal videos are labeled 0.  This will be used in custom_objective to keep track of normal and abnormal videos indexes.
            elif iv >= th_loop1:
                AllLabels[iv] = int(1)   # All instances of Normal videos are labeled 1. This will be used in custom_objective to keep track of normal and abnormal videos indexes.
    return  AllFeatures,AllLabels


#########################################################################################################
# MAIN CODE

adagrad=Adagrad(lr=0.001)

model.compile(loss=custom_objective, optimizer=adagrad)

print("Starting training...")

AllClassPath='NEW_UCSD_PED1_STIP_106/'                              # path to folder containing normal and abnormal videos' features
output_dir='NEW_UCSD_PED1_output/'                                  # Output_dir is the directory where you want to save trained weights
weights_paths = output_dir + 'weights.mat'                      # weights.mat are the model weights that you will get after (or during) that training
model_path = output_dir + 'model.json'

All_class_files= listdir(AllClassPath)
All_class_files.sort()
loss_graph =[]
total_iterations = 0
time_before = datetime.now()


for it_num in range(num_iters):

    AbnormalPath = os.path.join(AllClassPath, All_class_files[0])  # Path of abnormal already computed C3D features
    NormalPath = os.path.join(AllClassPath, All_class_files[1])    # Path of Normal already computed C3D features
    inputs, targets=load_dataset_Train_batch(AbnormalPath, NormalPath)  # Load normal and abnormal video C3D features

    batch_loss =model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1

    if total_iterations % 200 == 1:
        print "These iteration=" + str(total_iterations) + ") took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss)

weights_path = output_dir + 'weightsAnomalyL1L2_' + str(total_iterations) + '.mat'
save_model(model, model_path, weights_path)
