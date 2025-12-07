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

# torch.manual_seed(2)  # reproducible torch:2 np:3
# np.random.seed(3)

from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder
#from stream_BRICS import BIN_Data_Encoder

from math import sqrt
from scipy import stats

import pickle

from sklearn.model_selection import train_test_split

from sklearn.metrics import auc, precision_recall_curve


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

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
parser.add_argument('--task', choices=['davis', 'bindingdb', 'test'],
                    default='test', type=str, metavar='TASK',
                    help='Task name. Could be bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=0.00025, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
parser.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')

seed = 4221
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_task(task_name):
    if task_name.lower() == 'davis':
        return './dataset/Davis'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'test':
        return './dataset/Test'

def regression_scores(label, pred):
    label = torch.tensor(label).reshape(-1)
    pred = torch.tensor(pred).reshape(-1)
    rmse = sqrt(((label - pred) ** 2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)


def get_cindex(Y, P):
    P = P[:, np.newaxis] - P
    P = np.float32(P == 0) * 0.5 + np.float32(P > 0)
    Y = Y[:, np.newaxis] - Y
    Y = np.tril(np.float32(Y > 0), 0)
    P_sum = np.sum(P * Y)
    Y_sum = np.sum(Y)
    if Y_sum == 0:
        return 0
    else:
        return P_sum / Y_sum


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))




thresholds = [7.0]


def get_aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    roc_aupr = auc(recall, precision)
    return roc_aupr


def gen(y_true, y_pred):
    y_pred = torch.tensor(y_pred).reshape(-1)
    y_true = torch.tensor(y_true).reshape(-1)
    # Calculate metrics
    mse_loss = ((y_true - y_pred) ** 2).mean(axis=0)
    concordance_index = get_cindex(y_true, y_pred)
    rm2_value = get_rm2(y_true, y_pred)
    # rms_error = rmse(y_true, y_pred)
    # Calculate AUC values for each threshold
    # auc_values = [get_aupr((y_pred.cpu() > threshold).int(), y_true.view(-1, 1).float().cpu()) for threshold in thresholds]
    auc_values = [
        get_aupr((y_pred.cpu() > threshold).int(), y_true.view(-1, 1).float().cpu())
        for threshold in thresholds
    ]
    # Print the results
    print(f'MSE: {mse_loss:.4f}, CI: {concordance_index:.4f}, RM2: {rm2_value:.4f}')
    # print(f'RMS Error: {rms_error}')
    print(f'AUC Values: {auc_values}')


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        # m = torch.nn.Sigmoid()
        logits = torch.squeeze(score)
        # loss_fct = torch.nn.MSELoss()
        logits = torch.clamp(logits, min=5.0,
                             max=11.0)  ###BindingDB：min=0.0, max=15.0

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        # loss = loss_fct(logits, label)

        # loss_accumulate += loss
        # count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
    # loss = loss_accumulate / count
    gen(y_label, y_pred)
    rmse, pearson, spearman = regression_scores(y_label,
                                                y_pred)
    return rmse, pearson, spearman



def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0,
                           amsgrad=True)


    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True}

    dataFolder = get_task(args.task)

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # early stopping
    model_max = copy.deepcopy(model)
    best_res = 2.00 ** 10
    max_epo = 0


    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True

    for epo in range(args.epochs):
        '''if epo < 5:
            # 更新学习率 = 初始学习率 × 倍增系数
            for param_group in opt.param_groups:
                param_group['lr'] = 0.00002 *  (epo + 1)'''
        '''if epo == 5:
            for param_group in opt.param_groups:
                param_group['lr'] = args.lr'''
        model.train()
        s_epoch = time()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = nn.MSELoss()  # loss_fct = torch.nn.BCELoss()
            ###loss_fct = nn.L1Loss() # Mean Absolute Error, MAE
            # m = torch.nn.Sigmoid() ############softmax?
            n = torch.squeeze(score)  # n = torch.squeeze(m(score))

            loss = loss_fct(n, label)

            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))
                print(n)

        # every epoch test
        # '''
        with torch.set_grad_enabled(False):
            ###rmse_train, pearson_train, spearman_train = regression_scores(label, n)
            rmse_train, pearson_train, spearman_train = test(training_generator, model)
            rmse_val = rmse_train
            # rmse_val, pearson_val, spearman_val = test(validation_generator, model)
            rmse_test, pearson_test, spearman_test = test(testing_generator, model)
            if rmse_val < best_res:
                model_max = copy.deepcopy(model)
                best_res = rmse_val
                max_epo = epo
        # '''
        '''
        with torch.set_grad_enabled(False):
            # rmse_train, pearson_train, spearman_train = regression_scores(label, n)
            mse_train, ci_train, rm2_train = test(training_generator, model)
            mse_val, ci_val, rm2_val = test(validation_generator, model)
            mse_test, ci_test, rm2_test = test(testing_generator, model)
            if mse_val < best_res:
                model_max = copy.deepcopy(model)
                best_res = mse_val
        print('Train mse:{}, ci:{}, rm2:{}'.format(mse_train, ci_train, rm2_train))
        print('Validation mse:{}, ci:{}, rm2:{}'.format(mse_val, ci_val, rm2_val))
        print('Test mse:{}, ci:{}, rm2:{}'.format(mse_test, ci_test, rm2_test))
        '''
        e_epoch = time()
        print(e_epoch - s_epoch)
        # scheduler.step()  ##########################

    # 保存最优模型
    print(
        "Save the model:")  ###########################################################################################Save
    with open('Test.pkl', 'wb') as f:
        pickle.dump(model_max, f)
    print('Save best model from epo' + str(max_epo + 1))

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            rmse_test, pearson_test, spearman_test = test(testing_generator, model_max)
            print('Finally test result of rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test,
                                                                                   spearman_test))
            # rmse_test, pearson_test, spearman_test = test1(testing_generator, model_max)
            # print('Finally test result of rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test))
            # mse_test, ci_test, rm2_test = test(testing_generator, model_max)
            # print('Finally test result of mse:{}, ci:{}, rm2:{}'.format( mse_test, ci_test, rm2_test))

    except:
        print('testing failed')
    return model_max, loss_history

s = time()
model_max, loss_history = main()
e = time()
print(e - s)
