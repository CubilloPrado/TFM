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
logger = logging.getLogger('SAAM.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data_folder', default='data_', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best', help='Optional, name of the file in --model_dir containing weights to reload before training')  # 'best' or 'epoch_#'
parser.add_argument('--overlap', default=False, action='store_true', help='If we overlap prediction range during sampling')
parser.add_argument('--iterations-b-evaluations', default=5000, help='Whether to sample during evaluation')
parser.add_argument('--eres', default='evaluation_results.csv', help='Donde guardar el archivo')

def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    model.eval()
    with torch.no_grad():
        plot_batch = np.random.randint(1)
        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)
        predictions = []

        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(0).to(params.device)
            v_batch = v.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device)
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device)
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)
            emb = model.getEmbedding(test_batch, id_batch).to(params.device)
            Rx = model.get_Rx(emb.permute(1, 0, 2)).to(params.device)
            PSD = model.get_FFT_(Rx).to(params.device)
            H = torch.Tensor().to(params.device)
            plotting = True if (i == plot_batch) else False
            alpha_list = torch.empty(size=(params.test_predict_start, test_batch.shape[1], params.lstm_layers * params.lstm_hidden_dim, params.filtering_window))
            FFT_l_list = torch.empty(size=(params.test_predict_start, params.filtering_window, test_batch.shape[1], params.lstm_layers * params.lstm_hidden_dim, 2))
            attentive_FFT_l_list = torch.empty_like(alpha_list)

            for t in range(params.test_predict_start):
                zero_index = (test_batch[t,:,0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    test_batch[t,zero_index,0] = mu[zero_index]
                mu, sigma, hidden, cell, H, _, alpha, FFT_l, attentive_FFT_l = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell, PSD, H)
                input_mu[:,t] = v_batch * mu
                input_sigma[:,t] = v_batch * sigma
                if plotting:
                    alpha_list[t] = alpha
                    FFT_l_list[t] = FFT_l
                    attentive_FFT_l_list[t] = attentive_FFT_l

            if sample:
                samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, PSD, sampling=True)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative=params.relative_metrics)
            else:
                sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, PSD)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative=params.relative_metrics)

            if i == plot_batch:
                n_FFTs_per_sequence = 10
                plot_t_idx = sorted(random.sample(range(params.test_predict_start), n_FFTs_per_sequence))
                if sample:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples, relative=params.relative_metrics)
                else:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, relative=params.relative_metrics)
                size = 10
                top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // size]
                chosen = set(top_10_nd_sample.tolist())
                all_samples = set(range(batch_size))
                not_chosen = np.asarray(list(all_samples - chosen))
                if batch_size < 100:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
                else:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
                if batch_size < 12:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
                else:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
                combined_sample = np.concatenate((random_sample_10, random_sample_90))

                # Devolver todos los valores de `labels` sin filtrar
                label_plot = labels.data.cpu().numpy()
                predict_mu = sample_mu.data.cpu().numpy()
                predict_sigma = sample_sigma.data.cpu().numpy()
                plot_mu = np.concatenate((input_mu.data.cpu().numpy(), predict_mu), axis=1)
                plot_sigma = np.concatenate((input_sigma.data.cpu().numpy(), predict_sigma), axis=1)
                plot_metrics = {_k: _v for _k, _v in sample_metrics.items()}

                combined_sample = np.expand_dims(combined_sample, -1)
                alpha_list = alpha_list.data.cpu().numpy()[plot_t_idx, combined_sample,:,:]
                FFT_l_list = FFT_l_list.data.cpu().numpy()[plot_t_idx, :, combined_sample,:]
                attentive_FFT_l_list = attentive_FFT_l_list.data.cpu().numpy()[plot_t_idx, combined_sample,:,:]
                Rx_orig = model.get_Rx(test_batch.permute(1, 0, 2))
                FFT_orig = model.get_FFT_(Rx_orig).to(params.device)

                return summary_metric, plot_mu, plot_sigma, label_plot, plot_metrics, alpha_list, FFT_l_list, attentive_FFT_l_list, FFT_orig

        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        metrics_string = '; '.join('{}: {:05.6f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric, plot_mu, plot_sigma, label_plot, plot_metrics, alpha_list, FFT_l_list, attentive_FFT_l_list, FFT_orig

if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.iterations_b_evaluations = int(args.iterations_b_evaluations)

    logger.info('Loading the datasets...')
    num_train = 500000

    train_set = ClosingPrice(args.data_folder, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
    test_set = ClosingPriceTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)

    params.num_class = train_set.seq_num


    cuda_exist = torch.cuda.is_available()  # use GPU is available

    if cuda_exist:
        params.device = torch.device('cuda')
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = net.Net(params)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, num_workers=4)

    print('Model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics, plot_mu, plot_sigma, label_plot, plot_metrics, alpha_list, FFT_l_list, attentive_FFT_l_list, FFT_orig = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    # Guardar resultados adicionales en CSV
    results = {
        'plot_mu': plot_mu,
        'plot_sigma': plot_sigma,
        'label_plot': label_plot,
        'alpha_list': alpha_list
    }
    results_df = pd.DataFrame({k: [v.tolist()] for k, v in results.items()})
    csv_save_path = os.path.join(model_dir, args.eres)
    results_df.to_csv(csv_save_path, index=False)