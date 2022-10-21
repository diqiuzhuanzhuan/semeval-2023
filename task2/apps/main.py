# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import argparse
from pathlib import Path
import pandas as pd
import os, re
import time
from typing import AnyStr, Dict, Union
from allennlp.common.params import Params
import pytorch_lightning as pl
from task2.data_man.conll_reader import ConllDataModule
from task2.modeling.model import NerModel
from task2.configuration.config import logging
from task2.configuration import config
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def parse_arguments():

    parser = argparse.ArgumentParser(description='run experiments')

    parser.add_argument('--data_module_type', type=str, default='baseline_data_module',help='')
    parser.add_argument('--dataset_type', type=str, default='baseline_dataset', help='')
    parser.add_argument('--model_type', type=str, default='baseline_ner_model', help='')
    parser.add_argument('--encoder_model', type=str, default='xlm-roberta-base', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--max_epochs', type=int, default=1, help='')
    parser.add_argument('--lang', type=str, default='English', help='')
    parser.add_argument('--gpus', type=int, default=-1, help='')
    
    args = parser.parse_args()

    return args 


def get_model_earlystopping_callback(monitor='val_f1', mode:Union['max', 'min']='max', min_delta=0.001):
    es_clb = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=3,
        verbose=True,
        mode=mode
        )
    return es_clb


def get_model_best_checkpoint_callback(dirpath='checkpoints', monitor='val_f1', mode:Union['max', 'min']='max'):
    bc_clb = ModelCheckpoint(
        filename='{{epoch}}-{{{}:.3f}}-{{val_loss:.2f}}'.format(monitor),
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode=mode
        )
    return  bc_clb


def save_model(trainer: pl.Trainer, default_root_dir=".", model_name='', timestamp=None):
    out_dir = default_root_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logging.info('Stored model {}.'.format(outfile))
    best_checkpoint = None
    for file in os.listdir(out_dir):
        if file.startswith("epoch"):
            best_checkpoint = os.path.join(out_dir, file)
            break
    return outfile, best_checkpoint

def load_model(model_class: pl.LightningModule, model_file, stage='test', **kwargs):
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = model_class.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage)
    model.stage = stage
    return model

def write_eval_performance(args: argparse.Namespace, eval_performance: Dict, out_file: Union[AnyStr, bytes, os.PathLike]):
    out_file = Path(out_file)
    if not out_file.parent.exists():
        out_file.parent.mkdir()
    json_data = dict()
    for key, value in args._get_kwargs():
        json_data[key] = [value]
    for key in eval_performance:
        json_data[key] = [eval_performance[key]]
    json_data = pd.DataFrame(json_data)
    if out_file.exists():
        data = pd.read_csv(out_file)
        json_data = pd.concat([data, json_data])
    json_data.to_csv(out_file, index=False)
    logging.info('Finished writing evaluation performance for {}'.format(out_file.as_posix()))
    
def get_best_value(checkpoint_file: AnyStr, monitor: AnyStr='val_f1'):
    pattern = r'{}=(.*)-'.format(monitor)
    val = re.findall(pattern, checkpoint_file)[0]
    return float(val)

def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor

def get_trainer(args):
    pl.seed_everything(42)
    callbacks = [get_model_earlystopping_callback(monitor='val_micro@F1'), get_model_best_checkpoint_callback(monitor='val_micro@F1')]

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices=args.gpu, max_epochs=args.max_epochs, callbacks=callbacks)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=callbacks)

    logging.info('Finished create a trainer.')
    return trainer


if __name__ == '__main__':
    args = parse_arguments()
    trainer = get_trainer(args)
    dm = ConllDataModule.from_params(Params({
        'type': args.data_module_type,
        'reader': Params({
            'type': args.dataset_type
        }),
        'lang': args.lang,
        'batch_size': args.batch_size
    }))

    params = Params({
        'type': args.model_type,
        'encoder_model': args.encoder_model
    })
    ner_model = NerModel.from_params(params=params)
    trainer.fit(model=ner_model, datamodule=dm)
    _, best_checkpoint = save_model(trainer, model_name=args.model_type)
    val_f1 = get_best_value(best_checkpoint, monitor='val_micro@F1')
    write_eval_performance(args, {'val_micro@F1': val_f1}, config.performance_log)
    argument_model = load_model(NerModel.by_name(args.model_type), model_file=best_checkpoint)
    trainer.test(argument_model, datamodule=dm)
