import argparse
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import SAAM.net as net
from evaluate_SAAM import evaluate
from dataloader_test import *
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import utils as utils
from tensorboardX import SummaryWriter
logger = logging.getLogger('SAAM.Train')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ele', help='Name of the dataset')
parser.add_argument('--path', default='', type=str, help='Time series data path')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='ele', help='Directory containing params.json')
parser.add_argument('--cuda-device', default='0', help='GPU device to use')
parser.add_argument('--save-weights', default=True, help='Whether to save best ND to param_search.txt')
parser.add_argument('--force-cpu', default=False, help='Whether to force cpu to run the code')
parser.add_argument('--iterations-b-evaluations', default=2000, help='Whether to sample during evaluation')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,help='Optional, name of the file in --model_dir containing weights to reload before training')  # 'best' or 'epoch_#'
parser.add_argument('--overlap',default=False, action='store_true',help='If we overlap prediction range during sampling')


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling =  args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.force_cpu = args.force_cpu
    params.iterations_b_evaluations = int(args.iterations_b_evaluations)
    params.plot_dir_gradients = os.path.join(params.plot_dir, 'gradients')
    params.plot_dir_filtering = os.path.join(params.plot_dir, 'filtering')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
        os.mkdir(params.plot_dir_gradients)
        os.mkdir(params.plot_dir_filtering)
    except FileExistsError:
        pass


    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info(torch.__version__)
    logger.info("TIME RIGHT NOW!: " + str(datetime.now()))
    logger.info('Loading the datasets...')

    num_train = 500000

    train_set = ClosingPrice(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
    test_set = ClosingPriceTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)

 
