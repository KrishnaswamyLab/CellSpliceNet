import argparse
import torch
import numpy as np
import os
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from splicetransformer_model import SpliceTransformer
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/comparisons/utils/')
from seed import seed_everything
from log_utils import log

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/cellsplicenet-main/')
from data.splicedata_dataloader import splicedata_dataloader
from args import argparser_fn

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def get_data(batch_size):
    dataset_type = '_singlereplicant-nocontrol-fold01'
    args = argparser_fn(dataset_type, batch_size)
    args.dataset_root = os.path.join(ROOT_DIR, 'dataset')

    data = splicedata_dataloader(args)
    data.setup()
    data.setup_hparams(args)
    return data

def train_epoch(model, data, num_workers, optimizer, loss_fn, device, max_iter):
    train_loader = data.train_dataloader(shuffle_bool=True, num_workers=num_workers)
    train_loss = 0
    y_true_arr, y_pred_arr = None, None

    for iter_idx, data_item in enumerate(tqdm(train_loader)):
        '''
        data_item[0] is metadata
        data_item[1] is [mRNA sequence, annotation]
        data_item[2] is [PSI, delta PSI]

        For most applications, we only need:
        data_item[1][0] (mRNA sequence)
        data_item[1][1] (annotation: padding/exon/intron/flanking = 0/1/2/3)

        Our goal: (mRNA sequence + annotation) --predict--> PSI.
        '''
        if max_iter is not None and iter_idx > max_iter:
            break

        sequence, annotation = data_item[1]
        coded_seq = torch.hstack((sequence[:, None, :], annotation[:, None, :]))
        coded_seq = coded_seq.float().to(device)
        y_true = data_item[2]['psi'].to(device)

        y_pred = torch.sigmoid(model(coded_seq)[:, :, 0])
        loss = loss_fn(y_true, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.flatten().detach().cpu().numpy()
            y_pred_arr = y_pred.flatten().detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
            y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

    train_loss /= len(train_loader)
    pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
    spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
    R2_score = r2_score(y_true_arr, y_pred_arr)
    return model, train_loss, pearson_R, spearman_R, R2_score

def val_epoch(model, data, num_workers, loss_fn, device):
    val_loader = data.valid_dataloader(shuffle_bool=False, num_workers=num_workers)
    val_loss = 0
    y_true_arr, y_pred_arr = None, None

    for data_item in val_loader:
        sequence, annotation = data_item[1]
        coded_seq = torch.hstack((sequence[:, None, :], annotation[:, None, :]))
        coded_seq = coded_seq.float().to(device)
        y_true = data_item[2]['psi'].to(device)

        y_pred = torch.sigmoid(model(coded_seq)[:, :, 0])
        loss = loss_fn(y_true, y_pred)

        val_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.flatten().detach().cpu().numpy()
            y_pred_arr = y_pred.flatten().detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
            y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

    val_loss /= len(val_loader)
    pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
    spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
    R2_score = r2_score(y_true_arr, y_pred_arr)
    return model, val_loss, pearson_R, spearman_R, R2_score

def test_model(model, data, num_workers, loss_fn, device):
    test_loader = data.test_dataloader(shuffle_bool=False, num_workers=num_workers)
    test_loss = 0
    y_true_arr, y_pred_arr = None, None

    for data_item in test_loader:
        sequence, annotation = data_item[1]
        coded_seq = torch.hstack((sequence[:, None, :], annotation[:, None, :]))
        coded_seq = coded_seq.float().to(device)
        y_true = data_item[2]['psi'].to(device)

        y_pred = torch.sigmoid(model(coded_seq)[:, :, 0])
        loss = loss_fn(y_true, y_pred)

        test_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.flatten().detach().cpu().numpy()
            y_pred_arr = y_pred.flatten().detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
            y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

    test_loss /= len(test_loader)
    pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
    spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
    R2_score = r2_score(y_true_arr, y_pred_arr)
    return model, test_loss, pearson_R, spearman_R, R2_score


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description='Entry point.')
    cmd_parser.add_argument('--max-epochs', default=20, type=int)
    cmd_parser.add_argument('--max-training-iters', default=512, type=int)
    cmd_parser.add_argument('--batch-size', default=64, type=int)
    cmd_parser.add_argument('--learning-rate', default=2e-5, type=float)
    cmd_parser.add_argument('--num-workers', default=4, type=int)
    cmd_parser.add_argument('--random-seed', default=1, type=int)

    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_data(cmd_args.batch_size)

    # Train SpliceTransformer from scratch.
    model = SpliceTransformer(in_channels=2).to(device)

    # Set up training tools.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    # Load the data.
    data = get_data(cmd_args.batch_size)

    log_file = os.path.join(ROOT_DIR, 'comparisons', 'results', 'SpliceTransformer', f'log_seed-{cmd_args.random_seed}.txt')
    model_save_path = os.path.join(ROOT_DIR, 'comparisons', 'results', 'SpliceTransformer', f'model_seed-{cmd_args.random_seed}.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    log(f'[SpliceTransformer] Training begins.', filepath=log_file)
    best_val_loss = np.inf
    for epoch_idx in tqdm(range(cmd_args.max_epochs)):
        model.train()
        model, loss, pearson_R, spearman_R, R2_score = train_epoch(model, data, cmd_args.num_workers, optimizer, loss_fn, device, cmd_args.max_training_iters)
        scheduler.step()
        log(f'Epoch {epoch_idx}/{cmd_args.max_epochs}: Training Loss {loss:.3f}, P.R. {pearson_R:.3f}, S.R. {spearman_R:.3f}, R^2 {R2_score:.3f}.',
            filepath=log_file)

        with torch.no_grad():
            model.eval()
            model, loss, pearson_R, spearman_R, R2_score = val_epoch(model, data, cmd_args.num_workers, loss_fn, device)
            log(f'Validation Loss {loss:.3f}, P.R. {pearson_R:.3f}, S.R. {spearman_R:.3f}, R^2 {R2_score:.3f}.',
                filepath=log_file)

            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(model.state_dict(), model_save_path)
                log('Model weights successfully saved.', filepath=log_file)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model, loss, pearson_R, spearman_R, R2_score = test_model(model, data, cmd_args.num_workers, loss_fn, device)
    log(f'\n\nTest Loss {loss:.3f}, P.R. {pearson_R:.3f}, S.R. {spearman_R:.3f}, R^2 {R2_score:.3f}.',
        filepath=log_file)
