# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import functools
import pytorch_lightning as pl
from allennlp.common.registrable import Registrable
from typing import Any, AnyStr
from transformers import AutoModel
import torch


class NerModel(Registrable, pl.LightningModule):

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

       
@NerModel.register('baseline_ner_model')  
class BaselineNerModel(NerModel):
    def __init__(
        self, 
        encoder_model: AnyStr='xlm-roberta-base',
        lr: float=1e-5,
        warmup_steps: int=1000        
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.lr = lr
        self.warmup_steps = warmup_steps
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        warmup_steps = self.warmup_steps
        def fn(warmup_steps, step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return 1.0

        def linear_warmup_decay(warmup_steps):
            return functools.partial(fn, warmup_steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

