# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from task2.data_man.meta_data import read_conll_item_from_file
import os, collections
from typing import AnyStr, Union


def analyze_badcase(label_file: Union[AnyStr, os.PathLike], pred_file: Union[AnyStr, os.PathLike]):
    label_item = read_conll_item_from_file(label_file)
    pred_item = read_conll_item_from_file(pred_file)
    if len(pred_item) != len(label_item):
        raise ValueError("the length is not equal.")
    stat_map = collections.defaultdict(int)
    for label, pred in zip(label_item, pred_item):
        if label.id != pred.id:
            raise ValueError('id is not the same')
        for tag, pred_tag in zip(label.labels, pred.tokens):
            if tag == pred_tag:
                continue
            if tag == 'O':
                key = 'O->' + pred_tag[2:]
                stat_map[key] += 1
            elif pred_tag == 'O':
                key = tag[:2] + '->O'
                stat_map[key] += 1
            else:
                key = tag[:2] + pred_tag[2:]
                stat_map[key] += 1
    return stat_map
        
if __name__ == "__main__":
    pass