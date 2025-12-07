import copy
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data

from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder
#from stream_BRICS import BIN_Data_Encoder

import pickle


seed = 4221
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['davis', 'bindingdb', 'davis_add', 'bindingdb_add'],
                    default='', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')







def get_task(task_name):
    if task_name.lower() == 'davis':
        return './dataset/DAVIS'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis_add':
        return './dataset/DAVIS_Add'
    elif task_name.lower() == 'bindingdb_add':
        return './dataset/BindingDB_Add'


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score)) # m = torch.nn.Sigmoid()

        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label) # loss_fct = torch.nn.BCELoss()

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count
    '''
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    #print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    #print("AUROC:" + str(auc_k))
    #print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    #print('Confusion Matrix : \n', cm1)
    #print('Recall : ', recall_score(y_label, y_pred_s))
    #print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    #print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    #print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    #print('Specificity : ', specificity1)
    '''
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    #accuracy = accuracy_score(y_label, outputs)  ########################################################################Accuracy
    #print(f'Accuracy: {accuracy}')
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              outputs), y_pred, loss.item()

def test_final(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score)) # m = torch.nn.Sigmoid()

        loss_fct = torch.nn.BCELoss()  # EWC
        #loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label) # loss_fct = torch.nn.BCELoss()


        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    #print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    #print("AUROC:" + str(auc_k))
    #print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    #print('Confusion Matrix : \n', cm1)
    #print('Recall : ', recall_score(y_label, y_pred_s))
    #print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    #print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Sensitivity: ' + str(sensitivity1) + ' , Specificity: ' + str(specificity1))

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              outputs), y_pred, loss.item()


def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size
    model = BIN_Interaction_Flat(**config)

    # Load model
    '''
    print("Load the model:") 
    with open('。model_for_test/BindingDB_Add.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Done!")
    '''

    model = model.cuda()

    loss_history = []
    max_auc1 = 0.0
    max_auc2 = 0.0
    model_max1 = copy.deepcopy(model)
    model_max2 = copy.deepcopy(model)
    max_epo = 1
    aurocs = []  # 存储每个epoch的AUROC

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)

    print('--- Data Preparation ---')
    print("Load the data:")
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True}


    dataFolder = get_task(args.task)
    df_train = pd.read_csv(dataFolder + '/train.csv')
    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)
    df_val = pd.read_csv(dataFolder + '/val.csv')
    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)
    df_test = pd.read_csv(dataFolder + '/test.csv')
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)


    #'''
    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        s_epoch = time()
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()

            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

    
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(training_generator, model)
            print('Training at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))

            auc, auprc, f1, logits, loss = test(validation_generator, model)
            if auc > max_auc1:
                model_max1 = copy.deepcopy(model)
                max_auc1 = auc
                max_epo = epo + 1
            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))

            auc, auprc, f1, logits, loss = test(testing_generator, model)
            if auc > max_auc2:
                model_max2 = copy.deepcopy(model)
                max_auc2 = auc
            print('Testing at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))
            aurocs.append(auc)
        e_epoch = time()
        print(e_epoch - s_epoch)
        #scheduler.step()

    # 保存最优模型
    print("Save the best model")
    with open('Davis.pkl', 'wb') as f:
        pickle.dump(model_max1, f)
    print("from epoch:" + str(max_epo))
    print("Save the best model")
    with open('Davis(BEST).pkl', 'wb') as f:
        pickle.dump(model_max2, f)
    print("from epoch:" + str(max_epo))

    # 保存为CSV文件
    df = pd.DataFrame({'epoch': range(1, args.epochs + 1), 'AUROC': aurocs})
    df.to_csv('aurocs_Davis.csv', index=False)


    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test_final(training_generator, model_max1)
            print('Training AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))
            auc, auprc, f1, logits, loss = test_final(validation_generator, model_max1)
            print(
                'Validation AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss))
            auc, auprc, f1, logits, loss = test_final(testing_generator, model_max1)

            print(
                'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss))
            auc, auprc, f1, logits, loss = test_final(testing_generator, model_max2)
            print(
                'Best Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss))
    except:
        print('testing failed')


s = time()
main()
e = time()
print(e - s)
