import os
import random
import sys

import numpy as np
import torch

from datetime import datetime
import torch.nn.functional as F

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, args, seed=None):
    '''
    Initialize the log file of by redirecting the output.
    '''
    global original_stdout, original_stderr, outfile

    if seed is not None:
        set_seed(seed)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    f = open(os.path.join(log_path, f"log_{'attack_'+ str(args.inject_persent) if args.inject_user else 'normal'}_{datetime.now().strftime('%Y%m%d%H%M')}.txt"), 'w')
    f = Unbuffered(f)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    outfile = os.path.join(log_path, f"log_{datetime.now().strftime('%Y%m%d%H%M')}.txt")

    sys.stderr = f
    sys.stdout = f

def restore_stdout_stderr():
    '''
    Restore output.
    '''
    global original_stdout, original_stderr, outfile

    sys.stdout = original_stdout
    sys.stderr = original_stderr

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
    
def calculate_f1(predict_tensor, label_tensor, threshold=0.5):
    # Flatten the tensors
    predict_tensor = predict_tensor.detach().flatten()
    label_tensor = label_tensor.detach().flatten()

    # Binarize the predictions using the threshold
    predict_tensor = (predict_tensor > threshold).float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (predict_tensor * label_tensor).sum().float()
    fp = (predict_tensor * (1 - label_tensor)).sum().float()
    fn = ((1 - predict_tensor) * label_tensor).sum().float()

    return tp.item(), fp.item(), fn.item()

def bpr_loss(pos_preds, neg_preds, alpha=0.01):
    if len(pos_preds.shape) == 1:
        pos_preds = pos_preds.view(-1, 1)
    if len(neg_preds.shape) == 1:
        neg_preds = neg_preds.view(-1, 1)

    diff = pos_preds - neg_preds

    bpr_loss = -torch.sum(torch.log(torch.sigmoid(diff)))

    mse_loss = F.mse_loss(pos_preds, torch.ones_like(pos_preds), reduction='sum')

    loss = bpr_loss + alpha * mse_loss

    return loss


def custom_loss(scores, labels, alpha=0.05, beta=1.0):
    labels = labels.unsqueeze(-1).float().to(scores.device)
    
    pos_weight = torch.tensor(beta).to(scores.device)
    neg_weight = torch.tensor(1.).to(scores.device)
    
    weights = torch.where(labels == 1, pos_weight, neg_weight)
    
    bce_loss = F.binary_cross_entropy(scores, labels, weight=weights, reduction='none')
    entropy = -scores * torch.log(scores + 1e-10) - (1 - scores) * torch.log(1 - scores + 1e-10)
    entropy_regularization = torch.where(labels == 0, entropy, torch.zeros_like(entropy))
    total_loss = bce_loss + alpha * entropy_regularization
    
    return total_loss.mean()

def slice_lists(list1, list2, batch_size):
    len1, len2 = len(list1), len(list2)
    num_batches = -(-max(len1, len2) // batch_size)
    
    slice_size1 = -(-len1 // num_batches)
    slice_size2 = -(-len2 // num_batches)
    
    slices1 = [list1[i:i + slice_size1] for i in range(0, len1, slice_size1)]
    slices2 = [list2[i:i + slice_size2] for i in range(0, len2, slice_size2)]
    
    while len(slices1) < num_batches:
        slices1.append([])
    while len(slices2) < num_batches:
        slices2.append([])

    return slices1, slices2