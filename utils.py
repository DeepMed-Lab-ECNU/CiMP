

import torch
import torch
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = 1
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    Y = Y.type(torch.int64)
    c = c.type(torch.int64)
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
  
    uncensored_loss = -(c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - (1-c) * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss

    return loss


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_scheduler(optimizer, config, train_loader):
    if config.TRAIN.LR_SCHEDULER.NAME == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
            config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    elif config.TRAIN.LR_SCHEDULER.NAME == 'Cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(config.TRAIN.EPOCHS * len(train_loader)),
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=int(config.TRAIN.WARMUP_EPOCHS * len(train_loader)),
            cycle_limit=1,
            t_in_epochs=False,
        )

    return lr_scheduler


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou


class MetricLogger(object):

    def __init__(self, config):
        self.best_metirc = 0
        self.metric_list = []

        self.patience = 0
        self.max_patience = config.TRAIN.PATIENCE
        self.threshold = config.TRAIN.THRESHOLD
        self.stop_flag = False
        self.save_path = f"{config.CHECKPOINTS_PATH}/{config.CHECKPOINTS_NAME}"

    def early_stop(self, metric, model):
        self.patience += 1

        if metric >= self.best_metirc:
            self.best_metirc = metric
            self.patience = 0
            torch.save(model.state_dict(), self.save_path)

        if self.patience >= self.max_patience:
            self.stop_flag = True

    def cal_metric(self, mode='mean'):
        np_metric_list = np.array(self.metric_list)
        self.metric_list = []

        if mode == 'sum':
            return np_metric_list.sum()
        else:
            return np_metric_list.mean(), np_metric_list.std()
        

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix


def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]
    
    R_mat = torch.FloatTensor(R_mat).cuda()
 
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def CIndex_lifeline(hazards, labels, survtime_all):
    return concordance_index(survtime_all, hazards, labels)

def regularize_weights(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    return l1_reg


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value

    return pvalue_pred


def accuracy_cox(hazardsdata, labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)

    return correct / len(labels)


def CoxSurvLoss(hazards, S, c):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).cuda()
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg


def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model)
   
    return l1_reg
