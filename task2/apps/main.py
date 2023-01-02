# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os, re, sys, json
import time
from typing import AnyStr, Dict, List, Union
from tqdm import tqdm
from allennlp.common.params import Params
import pytorch_lightning as pl
from task2.data_man.conll_reader import ConllDataModule
from task2.data_man.meta_data import read_conll_item_from_file, write_conll_item_to_file
from task2.modeling.model import NerModel
from task2.configuration.config import logging
from task2.configuration import config
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from task2.data_man.badcases import analyze_badcase
from sklearn.model_selection import KFold

def parse_arguments():

    parser = argparse.ArgumentParser(description='run experiments')

    parser.add_argument('--data_module_type', type=str, default='baseline_data_module',help='')
    parser.add_argument('--dataset_type', type=str, default='span_aware_dataset', help='')
    parser.add_argument('--model_type', type=str, default='baseline_ner_model', help='')
    parser.add_argument('--encoder_model', type=str, default='xlm-roberta-base', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--max_epochs', type=int, default=1, help='')
    parser.add_argument('--lang', type=str, default='Chinese', help='')
    parser.add_argument('--monitors', type=str, default='val_micro@F1', help='the monitors you care about, use space as delimiter if many')
    parser.add_argument('--gpus', type=int, default=-1, help='')
    parser.add_argument('--cross-validation', type=int, default=1, help='make kfold cross validation')
    
    args = parser.parse_args()

    return args 

def k_fold(k: int=10, lang: AnyStr='English'):
    if k == 0:
        raise ValueError('k should be larger than 0')
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    if not config.kfold_data_path.exists():
        config.kfold_data_path.mkdir()
    items = read_conll_item_from_file(config.train_file[lang]) + read_conll_item_from_file(config.validate_file[lang])
    index = 0
    for train, val in kf.split(items):
        train_file = config.kfold_data_path/'{}-train_{}.conll'.format(config.code_by_lang[lang], index)
        validation_file = config.kfold_data_path/'{}-dev_{}.conll'.format(config.code_by_lang[lang], index)
        index += 1
        train_items = np.array(items)[train]
        write_conll_item_to_file(train_file, train_items, lang)
        val_items = np.array(items)[val]
        write_conll_item_to_file(validation_file, val_items, lang)
        yield train_file, validation_file


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

def write_test_results(test_results: List, out_file: Union[AnyStr, bytes, os.PathLike]):
    out_file = Path(out_file)
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True)
    with open(str(out_file), 'w') as f:
        for id, item in test_results:
            f.write(id+"\n")
            [f.write(line+"\n") for line in item]
            f.write("\n")

def write_stat_results(stat_dict: Dict, out_file: Union[AnyStr, bytes, os.PathLike]):
    out_file = Path(out_file)
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True)
    with open(str(out_file), 'w') as f:
        json.dump(stat_dict, f)
    logging.info('write stat file: {} done...'.format(str(out_file)))

            
def generate_result_file_parent(trainer: pl.Trainer, args: argparse.Namespace, value_by_monitor: Dict):
    parent_name = Path("_".join(["{}={}".format(k, v) for k, v in args._get_kwargs()]))/('version_'+str(trainer.logger.version))
    name = "{}={}".format(args.monitor, str(value_by_monitor[args.monitor])) + ".tsv"
    return parent_name, name

def validate_model(trainer: pl.Trainer, model: NerModel, data_module: pl.LightningDataModule):
    val_results = []
    preds = trainer.predict(model=model, dataloaders=data_module.val_dataloader())
    for id, tag_result in preds:
        val_results.extend(list(zip(*(id, tag_result))))
    return val_results

def test_model(trainer: pl.Trainer, model: NerModel, data_module: pl.LightningDataModule):
    test_results = []
    preds = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    for id, tag_result in preds:
        test_results.extend(list(zip(*(id, tag_result))))
    return test_results
    
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
        trainer = pl.Trainer(accelerator='gpu', devices=args.gpus, max_epochs=args.max_epochs, callbacks=callbacks)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=callbacks)

    logging.info('Finished create a trainer.')
    return trainer

def show_args(args):
    logging.info('run with these args:')
    log_info = "\n" + "\n".join(['{}: {}'.format(k, v) for k, v in args._get_kwargs()])
    logging.info(log_info)

def main(args: argparse.Namespace, train_file: AnyStr, val_file: AnyStr):
    trainer = get_trainer(args)
    dm = ConllDataModule.from_params(Params({
        'type': args.data_module_type,
        'reader': Params({
            'type': args.dataset_type,
            'encoder_model': args.encoder_model,
            'lang': args.lang
        }),
        'lang': args.lang,
        'batch_size': args.batch_size,
        'train_file': train_file,
        'val_file': val_file
    }))

    params = Params({
        'type': args.model_type,
        'encoder_model': args.encoder_model
    })
    ner_model = NerModel.from_params(params=params)
    trainer.fit(model=ner_model, datamodule=dm)
    _, best_checkpoint = save_model(trainer, model_name=args.model_type)
    ner_model = load_model(NerModel.by_name(args.model_type), model_file=best_checkpoint)
    logging.info('recording predictions of validation file....')
    parent, file = generate_result_file_parent(trainer, args, value_by_monitor)
    val_results = validate_model(trainer, ner_model, dm)
    out_file = config.output_path/parent/'val/'/file
    write_test_results(test_results=val_results, out_file=out_file)
    value_by_monitor = ner_model.get_metric()
    write_eval_performance(args, value_by_monitor, config.performance_log)
    out_file = config.output_path/parent/'metrics.tsv'
    write_eval_performance(args, value_by_monitor, out_file)

    test_results = test_model(trainer, ner_model, dm)
    out_file = config.output_path/parent/file
    write_test_results(test_results=test_results, out_file=out_file)
    stat_dict = analyze_badcase(label_file=config.test_file[args.lang], pred_file=out_file)
    stat_out_file = str(out_file) + ".stat.json"
    write_stat_results(stat_dict=stat_dict, out_file=stat_out_file)

if __name__ == "__main__":
    args = parse_arguments()
    show_args(args)
    if args.cross_validation == 1:
        main(args, config.train_file[args.lang], config.validate_file[args.lang])
    else:
        for train_file, val_file in k_fold(args.cross_validation, args.lang):
            main(args, train_file, val_file)

    sys.exit(0)