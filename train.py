import argparse
import random
import logging
import os
import json
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from pytorch_pretrained_bert import BertForTokenClassification

from data_loader import DataLoader
import utils
import numpy as np
from metrics import f1_score


def train(epoch, model, data_iterator, optimizer, scheduler, params):
    model.train()
    scheduler.step()

    t = trange(params.train_steps)
    for batch in t:
        batch_data, batch_tags = next(data_iterator)
        print(len(batch_data))
        batch_masks = batch_data.gt(0)
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)
        optimizer.step()

        if batch % 10 == 0:
            print('epoch:{}  batch: {}  train_loss: {:06.5f} '.format(epoch, batch, loss.item()))


def evaluate(model, data_iterator, params):
    model.eval()
    idx2tag = params.idx2tag
    true_tags = []
    pred_tags = []
    total_loss = 0.
    total_nums = 0

    for _ in range(params.eval_steps):
        batch_data, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)

        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        total_loss += loss.item()
        total_nums += 1

        batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # b, seq_len, num_labels
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()

        pred_tags.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in batch_tags for idx in indices])
    assert len(pred_tags) == len(true_tags)

    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = total_loss / total_nums
    metrics['f1'] = f1
    print('### evaluate_loss: {}   f1: {} ###'.format(metrics['loss'], metrics['f1']))
    return metrics


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir):
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        print("Epoch {}/{}".format(epoch, params.epoch_num))
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        train(epoch, model, train_data_iterator, optimizer, scheduler, params)
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, val_data_iterator, params)

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("### Best val f1: {:05.2f} ###".format(best_val_f1))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/msra', help="Directory containing the dataset")
    parser.add_argument('--bert_model_dir', default='bert_model', help="Directory of the BERT model")
    parser.add_argument('--model_dir', default='experiments/', help="Directory containing params.json")
    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")
    parser.add_argument('--restore_file', default=None, help="reload before training")
    parser.add_argument('--multi_gpu', default=False, help="Whether to use multiple GPUs")
    args = parser.parse_args()

    json_path = os.path.join('data', 'params.json')
    params = utils.Params(json_path)
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu
    params.seed = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------- load data -----------------------------------------
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params)
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    # -------------------------------- model -------------------------------------------------
    model = BertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=7)
    model.to(params.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
    # ----------------------------------------------------------------------------------------

    logging.info("######### Starting training! #####")
    train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, args.model_dir)

