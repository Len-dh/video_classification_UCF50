import numpy as np
import sys
import matplotlib.pyplot as plt 
import os 
import cv2
import pandas as pd
import random
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import sklearn
import copy
import torchvision.models as models
import random

from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from glob import glob
from PIL import Image
from sklearn import preprocessing
from sklearn.utils import compute_class_weight
from set_data.hmdb_dataset import hmdb_dataset
from models.video_models import Cnn_GRU, Cnn_lstm, GRU, resnet3D
from utils.display_max_pred_and_index_for_models import max_pred_and_idx
from utils.create_run_dir import create_run_dir
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# from pytorchtools import EarlyStopping


if __name__ ==  '__main__':
    import os    
    import pandas as pd

    NB_FOLDS = 5
    VIDEO_CUT_LENGTH = 80
    NB_EPOCHS = 100
    BATCH_SIZE = 2
    BASE_DIR = '/home/lenny/Bureau/Données_UCF/'
    DATABASE_DIR = 'UCF50-4-classes/'
    DATAFRAME_DIR = 'dataframe_UCF50_4_classes/'
    LOG_DIR = BASE_DIR + 'LOG_UCF50'
    LR = 1e-5
    NUM_CLASSES = 4
    PATIENCE = 5
    model_name = "video_model_"
    model_F1_plot = "F1_"
    model_Cross_e_plot = "Corss_e_loss_"
    model_losses_stats = "Losses_stats"
    model_F1_scores_stats = "F1_scores_stats"


    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    run_dir = os.path.join(LOG_DIR, model_name + date_time)

    dataframe_path = os.path.join(os.path.join(BASE_DIR + DATAFRAME_DIR,"frames_dataframe_UCF50-4-Classes.pkl"))    
    dataframe = pd.read_pickle(dataframe_path)
    print("size of dataframe : ", len(dataframe))

############################################## CNN model ################################################

    # choose a CNN model 

    # model_cnn = 'alexnet'
    model_cnn = 'vgg16'
    # model_cnn = 'vgg19'
    # model_cnn = 'resnet18'

    print("model is : ", model_cnn)

############################################ Implemented Model #############################################

    # choose an implemented model 

    # model_type = "resnet3D"
    model_type = "cnn_lstm"
    # model_type = "cnn_bi_lstm"
    # model_type = "Cnn_GRU"
    # model_type = "Cnn_bi_GRU"
    # model_type = "GRU"
    # model_type = "bi_GRU"

    print("model is : ", model_type)

    ## Stats for nb_group_by_class 

    label_list = []
    complete_video_names = []
    sequence_groupby_groups_and_labels = dataframe.groupby(['label','group', 'complete_video_name']).indices #TODO regarder si on inverse label et groupe 
    for (label, group, complete_video_name), indices in sequence_groupby_groups_and_labels.items():
        label_list.append(label)
        complete_video_names.append(complete_video_name)

    print("Class appearance")
    print(np.unique(label_list, return_counts=True))

    print("Appearance of group by label : ")
    label_by_group = np.unique(label_list, return_counts=True)
    print("Appearance of cut by group : ")
    nb_cut_by_group = np.unique(complete_video_names, return_counts=True)
    
    print(label_by_group)
    print(nb_cut_by_group)

    # exit()

    # group_list = []
    # idx_for_label_list = []
    # idx_for_group_list = []
    # idx_for_video_name_list = []
    # for i in range(len(label_by_group[0])):
    #     idx_for_label_list.append(label_by_group[0][i])
    #     idx_for_group_list.append(label_by_group[1][i]) 
    # print(idx_for_label_list)
    # print(idx_for_group_list)


    sequence_groupby_groups_and_labels = dataframe.groupby(['label','group','video_name']).indices #TODO regarder si on inverse label et groupe 
    indice_list = []
    label_list = []
    video_list = []
    video_name_list = []
    video_names = []
    labels = []
    videos = []
    for (label, group, video_name), indices in sequence_groupby_groups_and_labels.items():
        # if label == 'Diving':
        #     break
        labels.append(dataframe.iloc[indices]['label'].values[0]) 
        videos.append(list(dataframe.iloc[indices]['path']))
        video = list(dataframe.iloc[indices]['path'])
        video = tuple(video)

        video_path = video[0]
        video_label = video_path[47:-(len(video_name)+14)] # 14 is nb_frame, .jpg and group
        video_group = video_path[49 + len(video_label) + len(video_name) -4:-13]
    
        print(video_label)
        print(video_group)

        if video_label == label and video_group == group:
            ## list for sort videos by indices with label and group 
            indice_label_and_group = label + '_' + group
            print(indice_label_and_group)
            indice_list.append(indice_label_and_group)
            label_list.append(label)
            video_list.append(video)
            print(video)



    print("size of data : ", len(indice_list) ,len(label_list), len(video_list))

    dictionary = {} #for a key label_by_group, store list of videos by the samme group and the same label
    for k, v in zip(indice_list, list(zip(video_list, label_list))):
        dictionary.setdefault(k, []).append(v)

    # print(dictionary)
    # print("")
    # print(dictionary.get("Billiards_g09"))

    video_of_every_group_by_label = []
    first_indice = indice_list[0]
    previous_label_in_indice = first_indice[:-4]
    # print("previous_label_in_indice : ", previous_label_in_indice)

    values = []

    indice_list = sorted(list(set(indice_list)))
    video_list = sorted(list(set(video_list)))
    label_list = sorted(list(set(label_list)))
    
    print(indice_list)
    print(label_list)

    # # with list of unit label and list of indices, store list of element from keys of dictionnary
    # for label in label_list:
    #     #Billards
    #     for indice in indice_list:
    #         #Billiards_g01
    #         label_in_indice = indice[:-4]
    #         if label_in_indice == label:
    #             # Billiards == Billiards:
    #             #store element by key 
    #             value = dictionary[indice]
    #             value = tuple(value)
    #             print(value, type(value))
    #             if previous_label_in_indice == label_in_indice:
    #                 values.append(value)
    #                 print("label_in_indice : ", label_in_indice, " and previous label in dice : ", previous_label_in_indice)
    #             else:
    #                 # remove duplicate value 
    #                 video_of_every_group_by_label.append(values)
    #                 # print("video_of_every_group_by_label : ",video_of_every_group_by_label)
    #                 print(len(video_of_every_group_by_label))
    #                 print("label_in_indice : ", label_in_indice, " and previous label in dice : ", previous_label_in_indice)
    #                 print(video_of_every_group_by_label)
    #                 values = []
    #                 value = dictionary[indice]
    #                 values.append(value)

    #         previous_label_in_indice = label_in_indice
    # video_of_every_group_by_label.append(values)

    
    
    # print(video_of_every_group_by_label[3], len(video_of_every_group_by_label))
    # print("")
    # print(video_of_every_group_by_label[3][0])
    # print(len(video_of_every_group_by_label[0]))
    # print()
    # # exit()

    for indice in indice_list:
        value = dictionary[indice]
        for v in value:
            video_of_every_group_by_label.append(v)

    print(video_of_every_group_by_label, len(video_of_every_group_by_label))

    videos_by_group, labels = list(zip(*video_of_every_group_by_label))


    print(labels, len(labels))


    shortened_videos = []
    sorted_videos = []
    clueless_video = []
    clueless_label = []
    main_labels = []

    data = zip(videos_by_group, labels)

    for video, label in data:
        video = sorted(video)
        # print(video[0])
        #print("\n video : ", video, "\n and type : ", type(video),"\n and size : ", len(video))
        if len(video)>VIDEO_CUT_LENGTH:
            step_size = len(video) // VIDEO_CUT_LENGTH
            #print("\n step_size : ", step_size)
            start_frame = len(video) % VIDEO_CUT_LENGTH
            #print("\n start_frame : ", start_frame)
            shortened_video = video[start_frame::step_size] 
            #print("\n shortened_video : ", shortened_video, "\n and type : ", type(shortened_video),"\n and size : ", len(shortened_video))
            assert len(shortened_video) == VIDEO_CUT_LENGTH
            shortened_videos.append(shortened_video)

            main_labels.append(label)

        else:
            clueless_video.append(video)
            clueless_label.append(label)
        # sorted_videos.append(video)

    # need to have the same size of video frames
    print("\nnb of shortened_videos : ", len(shortened_videos))
    videos = np.array(shortened_videos, dtype=object)
    print("\nlen of videos : ", len(videos))
    print("\nlen of labels : ", len(labels), type(labels))

    #Verify the way to applying labels for videos
    # for video in videos: 
        # print("\nmain videos : ", video[0])

    # print("\nmain labels : ", main_labels)

    labels = np.array(main_labels)

    print("\nlen of labels : ", len(labels), type(labels))
    print("\nlabel after remove : ", labels)

    str_labels = np.array(labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    # print(labels)
    # print("")
    print(str_labels)
    print("")
    # print(labels)

    print("\nAppearances of labels")
    print(np.unique(labels, return_counts=True))

    print(np.unique(str_labels, return_counts=True))

    print("\nnb labels")
    print(len(np.unique(labels, return_counts=True)[0]))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n", "Run the code in device = ", device, "\n")

#     #####################################  models ############################################


    if model_type == "Cnn_GRU":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "num_layers": 2,
            "hidden_size": 128,
            "model_cnn": model_cnn,
            "bidirectional": False,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = Cnn_GRU(params_model)  

    if model_type == "Cnn_bi_GRU":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "num_layers": 2,
            "hidden_size": 128,
            "model_cnn": model_cnn,
            "bidirectional": True,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = Cnn_GRU(params_model) 

    if model_type == "GRU":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "input_size": 224*224*3,
            "num_layers": 2,
            "hidden_size": 128,
            "dropout": 0,
            "bidirectional": False,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = GRU(params_model)

    if model_type == "bi_GRU":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "input_size": 224*224*3,
            "num_layers": 2,
            "hidden_size": 128,
            "dropout": 0,
            "bidirectional": True,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = GRU(params_model)

    if model_type == "cnn_lstm":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "dr_rate": 0.5,
            "pretrained" : False,
            "rnn_num_layers": 2,
            "rnn_hidden_size": 2048,
            "model_cnn": model_cnn,
            "bidirectional": True,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = Cnn_lstm(params_model)

    if model_type == "cnn_bi_lstm":
        params_model={
            "num_classes": NUM_CLASSES,
            "device": device,
            "dr_rate": 0.1,
            "pretrained" : True,
            "rnn_num_layers": 2,
            "rnn_hidden_size": 128,
            "model_cnn": model_cnn,
            "bidirectional": True,
            "VIDEO_CUT_LENGTH": VIDEO_CUT_LENGTH}
        model = Cnn_lstm(params_model)

    if model_type == "resnet3D":
        params_model={
            "pretrained": True,
            "progress": False}
        model = resnet3D(params_model)

    print(model)
    model.to(device)

    #model.classifier’s parameters will use a learning rate of 1e-3, and a momentum 
    # of 0.9 will be used for all parameters. 
    # But here, adamW is the idea behind L2 regularization or weight decay is that networks 
    # with smaller weights (all other things being equal) are observed to overfit less and 
    # generalize better. 
    # Website explanation: https://towardsdatascience.com/why-adamw-matters-736223f31b5d

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    # Sets the learning rate of each parameter group to the initial lr times a given function.
    lr_lambda = lambda epoch : np.power(0.5, int(epoch/25)) # TODO:
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# # =============================================================================
# # Training and validation loop
# # =============================================================================
    #for FOLD in range(NB_FOLDS):     
    FOLD = 1
    run_name = model_name + "fold_" + str(FOLD)
    run_name_plt_F1 = model_F1_plot + "fold_" + str(FOLD)
    run_name_plt_Cross_e = model_Cross_e_plot + "fold_" + str(FOLD)
    run_name_plt_F1_scores_stats = model_F1_scores_stats + "fold_" + str(FOLD)
    run_name_plt_losses_stats = model_losses_stats + "fold_" + str(FOLD)

    ## explication sur le site : https://towardsdatascience.com/how-to-train-val-split-kfold-vs-stratifiedkfold-281767b93869
    random_number = 1   
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_number)

    ## Allow to split train and val videos list
    train_splits,test_splits = [],[]
    for train_index,test_index in skf.split(videos, labels):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_splits.append((list(videos[train_index]),list(labels[train_index])))
        test_splits.append((list(videos[test_index]),list(labels[test_index])))

    test_sequences,test_labels =test_splits[FOLD]
    train_sequences, train_labels = train_splits[FOLD]

    print(len(test_sequences))

    random_number = 1   
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_number)

    ## Allow to split train and val videos list
    train_splits,val_splits = [],[]
    for train_index,val_index in skf.split(train_sequences, train_labels):
        print("TRAIN:", train_index, "TEST:", val_index)
        train_splits.append((list(videos[train_index]),list(labels[train_index])))
        val_splits.append((list(videos[val_index]),list(labels[val_index])))

    train_sequences, train_labels = train_splits[FOLD]
    val_sequences, val_labels = val_splits[FOLD]

    print(len(train_sequences), len(val_sequences[0]), len(test_sequences))


    ## Calculated the ratio for each class 
    classes, class_appearances = np.unique(train_labels, return_counts=True)
    print("Train : Classes {} appear: {} times.".format(classes, class_appearances))

    classes_v, class_appearances_v = np.unique(val_labels, return_counts=True)
    print("Val : Classes {} appear: {} times.".format(classes_v, class_appearances_v))

    classes_test, class_appearances_test = np.unique(test_labels, return_counts=True)
    print("Test : Classes {} appear: {} times.".format(classes_test, class_appearances_test))

    ## Class_Weight of sklearn : https://androidkt.com/how-to-use-class-weight-in-crossentropyloss-for-an-imbalanced-dataset/
    ## Explanation : https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
    class_weights_sk=compute_class_weight('balanced',np.unique(train_labels),train_labels)
    class_weights_sk=torch.tensor(class_weights_sk,dtype=torch.float)

    print("class_weight of skleanr : ", class_weights_sk)
    
    loss_function = nn.CrossEntropyLoss(weight=class_weights_sk).to(device)     

    # print(len(train_labels), type(train_labels))

    train_dataset = hmdb_dataset(train_sequences, train_labels)#,
                                        # transform = data_transform)

    test_dataset = hmdb_dataset(test_sequences, test_labels)
 
    val_dataset = hmdb_dataset(val_sequences,val_labels)


    train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=2,
                                drop_last=True)

    test_loader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=2,
                                drop_last=True)


    val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=5,
                                drop_last=True)

    print("size of train_loader : ",len(train_loader))
    print("size of test_loader : ",len(test_loader))
    print("size of val_loader : ",len(val_loader))

    # dataiter = iter(train_loader)
    # images, labels_iter, index = dataiter.next()
    # print(images[0][0].shape)

    # fig = plt.figure()
    # plt.imshow(images[0][0].view(224, 224, 3))
    # print(labels_iter[0])
    # plt.title(str(labels_iter[0]) + "index : " + str(index))
    # plt.show()

    # print(images[0][1].shape)

    # fig = plt.figure()
    # plt.imshow(images[0][1].view(224, 224, 3))
    # print(labels_iter[1])
    # plt.title(str(labels_iter[1])  + "index : " + str(index))
    # plt.show()


    nb_epochs = NB_EPOCHS

    F1_scores_stats = {
    'train': [],
    'val': [],
    'test': []
    }
    
    loss_stats = {
        'train': [],
        'val': [],
        'test': []
        }
    
    best_F1 = 0.0
    is_best = False
    
    train_F1_list = []
    val_F1_list = []
    train_loss_list = []
    val_loss_list = []
    test_F1_list = []
    test_loss_list = []
    epoch_list = []
    train_stats_list = []
    val_stats_list = []

    n_epochs_stop = 6
    epochs_no_improve = 0
    early_stop = False


    print('\nstart training ...\n')
    # create_run_dir(run_dir)

    # for train
    confusion_matrix_train_list = []
    classification_report_train_list = []
    labels_list_train_list = []
    pred_list_train_list = []

    # for test 
    classification_report_val_list = []
    labels_list_val_list = []
    pred_list_val_list = []
    confusion_matrix_val_list = []

    # cpt for early stopping
    cpt_if_best_FALSE = 0


    # writer = SummaryWriter(opt.outf)
    for epoch in range(1, nb_epochs+1):
    # for epoch in range(1,5):

        pred_list_train = []
        labels_list_train = []

        pred_list_val = []
        labels_list_val = []
        
        # TRAINING
        train_epoch_loss = 0
        train_epoch_F1 = 0
        
        # writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print("\nepoch %d/%d, learning rate %f, best validation F1 so far: %f\n" % (epoch, NB_EPOCHS,
                                                                            optimizer.param_groups[0]['lr'],
                                                                            best_F1))
        # TRAINING
        correct = 0
        total = 0
        nb_classes = NUM_CLASSES
        confusion_matrix_train = np.zeros((nb_classes, nb_classes))
        train_losses = []

        model.train()

        start = time.time()
        for i_train, [inputs, labels] in enumerate(train_loader): 
            print("\n ### epoch %d/%d ### Train %d/%d\n" % (epoch, NB_EPOCHS, i_train, len(train_loader)))            
            inputs, labels = inputs.to(device), labels.to(device)

            if model_type == "resnet3D": 
                inputs = inputs.permute((0,2,1,3,4))
            
            optimizer.zero_grad()
    
            print("\nshape inputs / inputs type : ", inputs[0].shape, ' / ', type(inputs)) 

            pred = model.forward(inputs)

            max_preds_elemt, max_preds_idx = max_pred_and_idx(pred)
            # print("\nmax_preds elemt/idx: ",max_preds_elemt, " / ", max_preds_idx)

            train_loss = loss_function(pred, labels)
            # print('train_loss : ', train_loss)
            pred = torch.argmax(pred, dim=1)
            # print('pred : ', pred)
            # print('labels : ', labels)

            train_F1_score = f1_score(labels.cpu(), pred.cpu(), average='micro') 
            # print('train_F1_score : ', train_F1_score)
            # print('F1 ####: ', train_F1_score)

            # backward
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_F1 += train_F1_score.item()

            print("labels : ", labels)
            print("item of element 0 : ", labels.data[0].item())
            print("item of element 1 : ", labels.data[1].item(), "\n")

            print("pred : ", pred)
            print("item of element 0 : ", pred.data[0].item())
            print("item of element 1 : ", pred.data[1].item(), "\n")

            #for training (used to confusion matrix with BATCH_SIZE = 2)
            labels_list_train.append(labels.data[0].item())
            labels_list_train.append(labels.data[1].item())
            pred_list_train.append(pred.data[0].item())
            pred_list_train.append(pred.data[1].item())

            print("labels_list_train : ", labels_list_train)
            print("pred_list_train : ", pred_list_train)

            # Compute de confusion matrix
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix_train[t.long(), p.long()] += 1 

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            # End of train
            if i_train == len(train_loader)-1:
                print('i_train == len(train_loader)')

                train_stats = '\nEpoch ' + str(epoch) + ': \n \
                        |Train Loss \t: '+ str(train_epoch_loss/len(train_loader)) + ' \
                        |Train F1 \t: '+ str(train_epoch_F1/len(train_loader))

                train_stats_list.append(train_stats)

                classification_report_train = classification_report(labels_list_train, pred_list_train)

                confusion_matrix_train_list.append(confusion_matrix_train)
                classification_report_train_list.append(classification_report_train)
                labels_list_train_list.append(labels_list_train)
                print('labels_t : ', labels_list_train_list)
                pred_list_train_list.append(pred_list_train)
                print('pred_t : ', pred_list_train_list)

                # print('\ntrain_loss : ', train_loss)
                # print('train_F1_score : ', train_F1_score)
            
                print('\nF1 of the network on train images: %d %%' % (100 * correct / total))  
 
        # VALIDATION    
        correct = 0
        total = 0
        nb_classes = NUM_CLASSES
        confusion_matrix_val = np.zeros((nb_classes, nb_classes))

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_F1 = 0
            
            model.eval()
            for i_val, [inputs, labels] in enumerate(val_loader):
                print("\n ### epoch %d/%d ### Val %d/%d\n" % (epoch, NB_EPOCHS, i_val, len(val_loader)))            

                inputs, labels = inputs.to(device), labels.to(device) 

                if model_type == "resnet3D": 
                    inputs = inputs.permute((0,2,1,3,4))

                # print("\nshape inputs / inputs type : ", inputs[0].shape, ' / ', type(inputs)) 
                # print("\nshape labels / labels type : ", labels[0].shape, ' / ', type(labels)) 

                pred = model.forward(inputs)
                # print("pred : ", pred)

                max_preds_elemt, max_preds_idx = max_pred_and_idx(pred)
                # print("\nmax_preds elemt/idx: ",max_preds_elemt, " / ", max_preds_idx)

                val_loss = loss_function(pred, labels)
                # print('val loss : ', val_loss)
                pred = torch.argmax(pred, dim=1)
                # print('pred : ', pred)
                # print('labels : ', labels)
                val_F1_score = f1_score(labels.cpu(), pred.cpu(), average='micro')
                # print('val_F1_score : ',val_F1_score)
                
                val_epoch_loss += val_loss.item()
                val_epoch_F1 += val_F1_score.item()


                # for val (used to confusion matrix with BATCH_SIZE = 2)
                labels_list_val.append(labels.data[0].item())
                labels_list_val.append(labels.data[1].item())
                pred_list_val.append(pred.data[0].item())
                pred_list_val.append(pred.data[1].item())

                # Compute de confusion matrix
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix_val[t.long(), p.long()] += 1 

                total += labels.size(0)
                correct += (pred == labels).sum().item()

                # end of val 
                if i_val == len(val_loader)-1:
                    print('i_val == len(val_loader)')

                    val_stats ='\nEpoch ' + str(epoch) + ': \n \
                            |Val Loss \t: ' + str(val_epoch_loss/len(val_loader)) + ' \
                            |Val F1 \t: ' + str(val_epoch_F1/len(val_loader))

                    val_stats_list.append(val_stats)

                    classification_report_val = classification_report(labels_list_val, pred_list_val)

                    confusion_matrix_val_list.append(confusion_matrix_val)
                    classification_report_val_list.append(classification_report_val)
                    labels_list_val_list.append(labels_list_val)
                    # print('labels : ', labels_list_val_list)
                    pred_list_val_list.append(pred_list_val)
                    # print('pred : ', pred_list_val_list)

                    # print('\nval_loss : ', val_loss)
                    # print('val_F1_score : ', val_F1_score)
                
                    print('\nF1 of the network on val images: %d %%' % (100 * correct / total))  

    
        end = time.time()    


        print("time execution of model : ", end - start, '\n') 

        print('')
        print('confusion matrix for training : ')
        print(confusion_matrix_train)
        print('confusion matrix o get the per-class accuracy : ')
        print(np.diag(confusion_matrix_train)/confusion_matrix_train.sum(1))
        print('')
        print('confusion matrix_valing : ')
        print(confusion_matrix_val)
        print('confusion matrix o get the per-class accuracy : ')
        print(np.diag(confusion_matrix_val)/confusion_matrix_val.sum(1))
        

        print('F1 of the network on val images: %d %%' % (100 * correct / total))  
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))    
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        F1_scores_stats['train'].append(train_epoch_F1/len(train_loader))
        print("\nF1_scores_stats['train'] : ", F1_scores_stats['train'])
        F1_scores_stats['val'].append(val_epoch_F1/len(val_loader))
        print("\nF1_scores_stats['val'] : ", F1_scores_stats['val'])

        # calculate best F1 for model checkpoint------------------
        train_F1 = train_epoch_F1/len(train_loader)
        val_F1 = val_epoch_F1/len(val_loader)


        print("val_F1 : ", val_F1)
        print("best_F1 : ", best_F1)
        print("is_best : ", is_best)

        is_best = val_F1 > best_F1
        best_F1 = max(val_F1, best_F1)

        train_loss_for_plt = train_epoch_loss/len(train_loader)
        val_loss_for_plt = val_epoch_loss/len(val_loader)

        train_F1_list.append(train_F1)
        print("train_F1_list : ", train_F1_list)
        val_F1_list.append(val_F1)
        print("val_F1_list : ", val_F1_list)
        
        train_loss_list.append(train_loss_for_plt)
        print("train_loss_list : ", train_loss_list)
        val_loss_list.append(val_loss_for_plt)
        print("val_loss_list : ", val_loss_list)
        epoch_list.append(epoch)
        print("epoch_list : ", epoch_list)

        print(f'\nEpoch {epoch+0:03}: \n \
                |Train Loss \t: {train_epoch_loss/len(train_loader):.5f} \
                |Train F1 \t: {train_epoch_F1/len(train_loader):.3f}\n \
                |Val Loss \t: {val_epoch_loss/len(val_loader):.5f} \
                |Val F1 \t: {val_epoch_F1/len(val_loader):.3f}')

        # binary classification
        #for train
        # print("\nlabels list for train : ", labels_list_train_list)
        print("\nsize of labels list for train : ", len(labels_list_train_list))
        # print("\npred list for train : ", pred_list_train_list)
        print("\nsize of pred list for train : ", len(pred_list_train_list))

        print('\nprecision_recall_fscore_support(labels,pred, average=None) : ',precision_recall_fscore_support(labels_list_train,pred_list_train, average=None))

        print(classification_report(labels_list_train, pred_list_train))

        #for val

        # print("\nlabels list for val : ", labels_list_val_list)
        print("\nsize of labels list for val : ", len(labels_list_val_list))
        # print("\npred list for val : ", pred_list_val_list)
        print("\nsize of pred list for val : ", len(pred_list_val_list))

        print('\nprecision_recall_fscore_support(labels,pred, average=None) : ',precision_recall_fscore_support(labels_list_val,pred_list_val, average=None))

        print(classification_report(labels_list_val, pred_list_val))

        # # multiclass 
        # print('\nprecision_recall_fscore_support(labels,pred, average=macro) : ',precision_recall_fscore_support(labels_list,pred_list)
        # print('\nprecision_recall_fscore_support(labels,pred, average=micro) : ',precision_recall_fscore_support(labels_list,pred_list, average='micro'))
        # print('\nprecision_recall_fscore_support(labels,pred, average=weighted) : ',precision_recall_fscore_support(labels_list,pred_list, average='weighted'))
      
        # print(classification_report(labels, pred))

        print("is_best : ", is_best)

        if is_best == False:
            cpt_if_best_FALSE +=1 
        else:
            cpt_if_best_FALSE = 0

        print("cpt_if_best_FALSE : ", cpt_if_best_FALSE)

        # early stopping
        if is_best == False and cpt_if_best_FALSE == PATIENCE:
            print("Early stopping ! The network is done :'(")
            break

        if is_best: 
            create_run_dir(run_dir)
            print("Saving best model with F1: {}".format(best_F1))
            best_model = copy.deepcopy(model)
            torch.save({
                'epoch': epoch,
                'best_model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_stats': loss_stats,
                'F1_scores_stats': F1_scores_stats,
                }, os.path.join(run_dir,run_name + '_best_model_checkpoint.pth.tar'))
    
    ## TEST

    checkpoint = torch.load(os.path.join(run_dir,run_name + '_best_model_checkpoint.pth.tar'))
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['best_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_stats = checkpoint['loss_stats']
    F1_scores_stats = checkpoint['F1_scores_stats']


    # for test 
    classification_report_test_list = []
    labels_list_test = []
    labels_list_test_list = []
    pred_list_test = []
    pred_list_test_list = []
    confusion_matrix_test_list = []

    # for matrix_confusion_test
    correct = 0
    total = 0
    nb_classes = NUM_CLASSES
    confusion_matrix_test = np.zeros((nb_classes, nb_classes))

    test_epoch_loss = 0
    test_epoch_F1 = 0
    with torch.no_grad():


        model.eval()
        for i_test, [inputs, labels] in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device) 

            if model_type == "resnet3D": 
                inputs = inputs.permute((0,2,1,3,4))

            pred = model.forward(inputs)

            max_preds_elemt, max_preds_idx = max_pred_and_idx(pred)

            test_loss = loss_function(pred, labels)
            pred = torch.argmax(pred, dim=1)
            test_F1_score = f1_score(labels.cpu(), pred.cpu(), average='micro')

            test_epoch_loss += test_loss.item()
            test_epoch_F1 += test_F1_score.item()

           # for val (used to confusion matrix with BATCH_SIZE = 2)
            labels_list_test.append(labels.data[0].item())
            labels_list_test.append(labels.data[1].item())
            pred_list_test.append(pred.data[0].item())
            pred_list_test.append(pred.data[1].item())

            # Compute de confusion matrix
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix_test[t.long(), p.long()] += 1 
            

            # end of test 
            if i_test == len(test_loader)-1:
                print('i_test == len(test_loader)')

                classification_report_test = classification_report(labels_list_test, pred_list_test)

                confusion_matrix_test_list.append(confusion_matrix_test)
                classification_report_test_list.append(classification_report_test)

        loss_stats['test'].append(test_epoch_loss/len(test_loader))
        F1_scores_stats['test'].append(test_epoch_F1/len(test_loader))
            
    print('classification report test : ', classification_report_test)
    print('confusion matrix test : ',confusion_matrix_test)


    fig3 = plt.figure()
    plt.plot(F1_scores_stats['train'], 'r', label = 'F1 score train')
    plt.plot(F1_scores_stats['val'], 'g', label = 'F1 score val')
    plt.legend(frameon=False)
    plt.show()
    fig3.savefig(os.path.join(run_dir, run_name_plt_losses_stats + '.png'))

    fig4 = plt.figure()
    plt.plot(loss_stats['train'], 'r', label = 'Training loss')
    plt.plot(loss_stats['val'], 'g', label = 'Validation loss')
    plt.legend(frameon=False)
    plt.show()
    fig4.savefig(os.path.join(run_dir, run_name_plt_F1_scores_stats + '.png'))

    model_type_extend = []
    model_type_extend.extend([model_type for i in range(len(epoch_list))])
    model_cnn_extend = []
    model_cnn_extend.extend([model_cnn for i in range(len(epoch_list))])
    time_execution = end - start
    time_execution_extend = []
    time_execution_extend.extend([time_execution for i in range(len(epoch_list))])
    LR_extend = []
    LR_extend.extend([LR for i in range(len(epoch_list))])
    size_of_dataframe = len(dataframe)
    size_of_dataframe_extend = []
    size_of_dataframe_extend.extend([size_of_dataframe for i in range(len(epoch_list))])
    nb_videos = len(videos)
    nb_videos_extend = []
    nb_videos_extend.extend([nb_videos for i in range(len(epoch_list))])
    class_appearances_train = "Train : Classes {} appear: {} times.".format(classes, class_appearances)
    class_appearances_train_extend = []
    class_appearances_train_extend.extend([class_appearances_train for i in range(len(epoch_list))])
    class_appearances_val = "Val : Classes {} appear: {} times.".format(classes_v, class_appearances_v)
    class_appearances_val_extend = []
    class_appearances_val_extend.extend([class_appearances_val for i in range(len(epoch_list))])
    class_appearances_test = "test : Classes {} appear: {} times.".format(classes_test, class_appearances_test)
    class_appearances_test_extend = []
    class_appearances_test_extend.extend([class_appearances_test for i in range(len(epoch_list))])
    model_name_extend = []
    model_name_extend.extend([model_name for i in range(len(epoch_list))])
    NUM_CLASSES_extend = []
    NUM_CLASSES_extend.extend([NUM_CLASSES for i in range(len(epoch_list))])
    filename_pth = os.path.join(run_dir,run_name + '_model_checkpoint.pth.tar')
    filename_pth_extend = []
    filename_pth_extend.extend([filename_pth for i in range(len(epoch_list))])
    img_F1_score_path = run_dir + run_name_plt_F1 + '.png'
    img_F1_score_path_extend = []
    img_F1_score_path_extend.extend([img_F1_score_path for i in range(len(epoch_list))])
    img_cross_e_path = run_dir + run_name_plt_Cross_e + '.png'
    img_cross_e_path_extend = []
    img_cross_e_path_extend.extend([img_cross_e_path for i in range(len(epoch_list))])

    test_stats = '|Test Loss \t: ' + str(test_epoch_loss/len(test_loader)) + ' \
    |Test F1 \t: ' + str(test_epoch_F1/len(test_loader))

    test_stats_list = []
    classification_report_test_list = []
    confusion_matrix_test_list = []
    labels_list_test_list = []
    pred_list_test_list = []
    for i in range(len(epoch_list)):
        test_stats_list.append(" ")
        classification_report_test_list.append(" ")
        confusion_matrix_test_list.append(" ")
        labels_list_test_list.append(" ")
        pred_list_test_list.append(" ")
        if epoch_list[-1]: 
            test_stats_list.append(test_stats)
            classification_report_test_list.append(classification_report_test)
            confusion_matrix_test_list.append(confusion_matrix_test)
            labels_list_test_list.append(labels_list_test)
            pred_list_test_list.append(pred_list_test)
    

    zipped_dataframe_metric =  list(zip(model_type_extend, model_cnn_extend, time_execution_extend, LR_extend, size_of_dataframe_extend,
    nb_videos_extend, class_appearances_train_extend, class_appearances_val_extend, class_appearances_test_extend,
    NUM_CLASSES_extend, labels_list_train_list, pred_list_train_list, labels_list_val_list, pred_list_val_list, labels_list_test_list, pred_list_test_list,
    filename_pth_extend, img_F1_score_path_extend, img_cross_e_path_extend, train_stats_list, val_stats_list, 
    classification_report_train_list, confusion_matrix_train_list, classification_report_val_list, confusion_matrix_val_list,classification_report_test_list, confusion_matrix_test_list, test_stats_list))

    dataframe_metric = pd.DataFrame(zipped_dataframe_metric, columns=['model_type', 'model_cnn', 'time execution of model', 'learning rate', 'nb_data_in_dataframe',
    'nb_videos','class appearances train','class appearances val', 'class appearances test', 
    'nb_classes', 'labels train', 'pred train','labels val', 'pred val', 'labels test', 'pred test',
    'filename_pth_igms_F and loss_function', 'img_F1_score', 'img_loss', 'train_stats', 'val_stats', 
    'classification report train','confusion matrix train', 'classification report val', 'confusion maxtrix val', 'classification report test', 'confusion matrix test', 'test_stats'], index = [ i for i in epoch_list])

    print(dataframe_metric)
    print('saving dataframe_metric .....')
    dataframe_metric_csv = dataframe_metric.to_csv(run_dir + '/dataframe_metric_' + date_time + '.csv')

    # save last model and complete statistics   
    print("saving statistics ...")    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_stats': loss_stats,
                'F1_scores_stats': F1_scores_stats
                }, filename_pth)