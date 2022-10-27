# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import functools
from importlib.metadata import metadata
from itertools import compress
import pytorch_lightning as pl
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from typing import Any, AnyStr
from transformers import AutoModelForTokenClassification
import torch
from task2.data_man.meta_data import get_num_labels, get_type_by_id, extract_spans, get_id_to_labes_map, is_id_legal
from task2.metric.span_metric import SpanF1
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions


def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def linear_warmup_decay(warmup_steps):
    return functools.partial(fn, warmup_steps)


class NerModel(Registrable, pl.LightningModule):
    lr = 1e-5
    warmup_steps = 1000

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        warmup_steps = self.warmup_steps

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
        

    def forward_step(self, batch):
        raise NotImplementedError()

       
@NerModel.register('baseline_ner_model')  
class BaselineNerModel(NerModel):

    def __init__(
        self, 
        encoder_model: AnyStr='xlm-roberta-base',
        lr: float=1e-5,
        warmup_steps: int=1000
        ) -> None:
        super().__init__()
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_model, num_labels=get_num_labels())
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.metric = SpanF1()
        self.save_hyperparameters({
            'encoder_model': encoder_model,
            'lr': lr,
            'warmup_steps': warmup_steps
        })

    def forward_step(self, batch: Any):
        id, input_ids, token_type_ids, attention_mask, token_masks, tag_lens, label_ids, gold_spans = batch
        if self.encoder.config.type_vocab_size < 2:
            token_type_ids = None
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=label_ids
        )
        preds = torch.argmax(outputs.logits, -1)
        return_dict = {
            'preds': preds
        }
        if label_ids is not None:
            return_dict['loss'] = outputs.loss
        return_dict.update(self._compute_token_tags(preds=preds, tag_lens=tag_lens, gold_spans=gold_spans))
        
        return return_dict

    def _compute_token_tags(self, preds: torch.Tensor, tag_lens: Any, gold_spans: Any):
        batch_size = len(preds)
        preds = preds.cpu().numpy()
        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_len = tag_lens[i]
            tag_seq = preds[i][:tag_len]
            pred_tags.append([get_type_by_id(x) for x in tag_seq])
            pred_results.append(extract_spans(pred_tags[-1]))
        output = {
            'token_tags': pred_tags
        }
        if gold_spans is not None:
            self.metric(pred_results, gold_spans)
            output["metric"] = self.metric.compute()

        return output 
    
    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.forward_step(batch)
        self.log_metrics(outputs['metric'], outputs['loss'], suffix='train_', on_step=True, on_epoch=False)
        return {'loss': outputs['loss']}

    def on_train_epoch_start(self) -> None:
        self.metric.reset()
        return super().on_train_epoch_start()

    def training_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.tensor([item['loss'] for item in outputs], device=self.device))
        metric = self.metric.compute()
        self.log_metrics(metric, average_loss, suffix='train_', on_step=False, on_epoch=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.forward_step(batch)
        self.log_metrics(outputs['metric'], outputs['loss'], suffix='val_', on_step=True, on_epoch=False)
        return {'loss': outputs['loss']}

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.tensor([item['loss'] for item in outputs], device=self.device))
        metric = self.metric.compute()
        self.log_metrics(metric, average_loss, suffix='val_', on_step=False, on_epoch=True)
        return super().validation_epoch_end(outputs)

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.forward_step(batch)
        pred_tags = outputs['token_tags']
        id, input_ids, token_type_ids, attention_mask, token_masks, label_ids, gold_spans = batch
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_masks)]
        return tag_results

    def test_epoch_end(self, outputs) -> None:
        
        return super().test_epoch_end(outputs)

    def predict_tags(self, batch: Any):
        return self.test_step(batch=batch, batch_idx=0)


@NerModel.register('baseline_crf_model') 
class BaselineCrfModel(BaselineNerModel): 

    def __init__(
        self,
        encoder_model: AnyStr = 'xlm-roberta-base', 
        lr: float = 2e-5, 
        warmup_steps: int = 3000
        ) -> None:
        super().__init__(encoder_model, lr, warmup_steps)
        self.crf_layer = ConditionalRandomField(
            num_tags=get_num_labels(),
            constraints=allowed_transitions(constraint_type="BIO", labels=get_id_to_labes_map())
            )

    def forward_step(self, batch: Any):
        id, input_ids, token_type_ids, attention_mask, token_masks, tag_lens, label_ids, gold_spans = batch
        if self.encoder.config.type_vocab_size < 2:
            token_type_ids = None
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=label_ids, 
            token_type_ids=token_type_ids,
            output_hidden_states=True
            )
        token_scores = outputs.logits
        #token_scores = torch.log_softmax(outputs.logits, dim=-1)  # try or not try
         
        batch_size = len(id)
        return_dict = dict()
        crf_mask = torch.ones_like(input_ids, device=self.device)
        for i in range(batch_size):
            crf_mask[tag_lens[i]:] = 0
        if label_ids is not None:
            label_ids[label_ids==-100] = 0
            loss = -self.crf_layer(token_scores, label_ids, crf_mask) / float(batch_size)
            return_dict['loss'] = loss
        best_path = self.crf_layer.viterbi_tags(token_scores, crf_mask)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_tags.append([get_type_by_id(x) for x in tag_seq])
            pred_results.append(extract_spans([get_type_by_id(x) for x in tag_seq if is_id_legal(x)]))
        return_dict['token_tags'] = pred_tags
        if gold_spans is not None:
            self.metric(pred_results, gold_spans)
            return_dict['metric'] = self.metric.compute()
        return return_dict 

    
if __name__ == "__main__":
    params = Params({
        'type': 'baseline_ner_model',
        'encoder_model': 'xlm-roberta-base'
    })
    model = NerModel.from_params(params=params)