import argparse
import torch
import os

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='LoRec')

parser.add_argument('--seed', type=int, default=2023, help='seed')

parser.add_argument('--model', type=str, default='SASrec', help='model')
parser.add_argument('--use_LLM', type=str2bool, default=True, help='training device')

# dataset
parser.add_argument('--dataset', type=str, default='Games', help='dataset')
parser.add_argument('--recommendation_scenario', type=str, default='Arts Recommendation', help='Recommendation scenario')
parser.add_argument('--max_interaction', type=int, default=50, help='Max interactions')

# model
parser.add_argument('--LLM', type=str, default='Llama2', help='LLM model')
parser.add_argument('--LLM_size', type=int, default=4096, help='Output size of LLM')
parser.add_argument('--out_dim', type=int, default=512, help='Output size of Adapter')

# experiment
parser.add_argument('--use_gpu', type=str2bool, default=True, help='training device')
parser.add_argument('--device', type=str, default='gpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--val_interval', type=int, default=10, help='Validation interval')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight for L2 loss on basic models.')
parser.add_argument('--min_epochs', type=int, default=150, help='min epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='max epoch')

parser.add_argument('--with_lct', type=str2bool, default=True, help='Whether use LCT to enhance RS Robustness')
parser.add_argument('--lct_start', type=int, default=30, help='Which epoch of LCT begins')
parser.add_argument('--lct_minibatch', type=int, default=2, help='How many times does LCT train in each epoch.')
parser.add_argument('--reg_entropy', type=float, default=0.5, help='lambada_1: weight of entropty regularization')
parser.add_argument('--sim_weight', type=float, default=1.0, help='lambada_2: weight of llm-enhanced')
parser.add_argument('--user_weight', type=float, default=3.0, help='Initial weight coefficient of each user')
parser.add_argument('--weight_update', type=int, default=5, help='How many times does FD train in each epoch.')

parser.add_argument('--top_k', type=int, default=10, help='K in evaluation')
parser.add_argument('--attack_top_k', type=int, default=50, help='K in evaluation of attack')

parser.add_argument('--inject_user', type=str2bool, default=True, help='Whether has injected users in dataset')
parser.add_argument('--inject_persent', type=str, default='random_1', help='type and presentage of injected user (attack file)')

parser.add_argument('--baseline', type=str, default='ADV', help='presentage of injected user')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.dataset == "MIND":
    args.recommendation_scenario = "News Recommendation"
elif args.dataset == "Games":
    args.recommendation_scenario = "Game Recommendation"
elif args.dataset == "Arts":
    args.recommendation_scenario = "Arts Recommendation"

if args.LLM == "Llama2":
    args.LLM_size = 4096
elif args.LLM == "Llama2_13":
    args.LLM_size = 5120
elif args.LLM == "Llama2_70":
    args.LLM_size = 8192

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'