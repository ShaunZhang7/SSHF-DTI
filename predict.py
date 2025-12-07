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
from models import BIN_Interaction_Flat  #############################################
from stream import BIN_Data_Encoder  #############################################

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
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
parser.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')


def regression_scores(label, pred):
    label = torch.tensor(label).reshape(-1)
    pred = torch.tensor(pred).reshape(-1)
    rmse = sqrt(((label - pred) ** 2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)

def get_cindex111(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])
    if pair != 0:
        return summ / pair
    else:
        return 0

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

def tefdta(y_true, y_pred):
    y_true = torch.tensor(y_true).reshape(-1)
    y_pred = torch.tensor(y_pred).reshape(-1)
    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    ci = get_cindex(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    return mse, ci, rm2

# thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]
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

    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        # m = torch.nn.Sigmoid()
        logits = torch.squeeze(score)
        # loss_fct = torch.nn.MSELoss()
        logits = torch.clamp(logits, min=0.0,
                             max=15.0)  ###Davis:min=5.0, max=11.0 BindingDB:min=0.0, max=15.0

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()


        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
    return y_pred


def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size
    model = BIN_Interaction_Flat(**config)
    # 加载Davis最优模型
    print("Load the model:") ###########################################################################################原模型
    with open('davis_dta_BRICS.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Done!")
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': True}


    df_train = pd.read_csv('./dataset/Predict/predict_example.csv')
    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)


    print('--- Go for Predicting ---')
    try:
        with torch.set_grad_enabled(False):
            predict = test(training_generator, model)

            # 确保predict列表的长度与df_train的行数相同
            assert len(predict) == len(df_train), "预测结果列表的长度必须与原始DataFrame的行数相同"

            # 将predict列表转换为Pandas Series，并命名为'predict'
            predict_series = pd.Series(predict, name='predict')

            # 将predict_series添加到df_train的最后一列
            df_train['predict'] = predict_series

            # 保存更新后的DataFrame为新的CSV文件
            df_train.to_csv('./dataset/Predict/result_example.csv', index=False)

    except:
        print('testing failed')


s = time()
main()
e = time()
print(e - s)
