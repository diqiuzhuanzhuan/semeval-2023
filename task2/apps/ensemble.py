# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import List, AnyStr
import numpy as np
import os
from sklearn.metrics import classification_report
import itertools
from task2.data_man.meta_data import get_id_by_type, get_type_by_id, LABEL_NAME, read_conll_item_from_file
from task2.configuration.config import logging
from pathlib import Path

def vote(dev_label_files: List[AnyStr], dev_files: List[AnyStr], test_files: List[AnyStr], output_file):
    assert len(dev_files) == len(test_files) == len(dev_label_files)
    def _get_dev_report(dev_file, dev_label_file):
        y_pred = [item.labels for item in read_conll_item_from_file(dev_file, just_label=True)]
        y_pred = list(itertools.chain(*y_pred))
        y_true = [item.labels for item in read_conll_item_from_file(dev_label_file)]
        y_true = list(itertools.chain(*y_true))
        length = min(len(y_true), len(y_pred))
        report_dict = classification_report(y_true=y_true[0:length], y_pred=y_pred[0:length], output_dict=True)
        report_dict['O'] = report_dict['macro avg']
        return report_dict

    def _calc_score(test_file, report_dict, final_res=[]):
        if not final_res:
            final_res = [[[0 for i in LABEL_NAME] for j in range(len(item.labels))] for item in read_conll_item_from_file(test_file, just_label=True)]
        for idx, item in enumerate(read_conll_item_from_file(test_file, just_label=True)):
            for jdx, tag in enumerate(item.labels):
                if tag == 'O':
                    _tag = tag
                else:
                    _tag = tag[2:]
                score = report_dict[tag]['f1-score']
                final_res[idx][jdx][get_id_by_type(tag)] += score
        return final_res

    def _weight_sum(final_res):
        final_y_pred = [[get_type_by_id(np.argmax(k)) for k in l] for l in final_res]
        return final_y_pred

    def _write_vote_result(output_file, final_y_pred):
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, "w") as f:
            f.write('\n')
            for l in final_y_pred:
                f.write('\n'.join(l))
                f.write("\n\n\n")
        
    final_res = []
    for dev_file, dev_label_file, test_file in zip(dev_files, dev_label_files, test_files):
        if not Path(dev_file).exists() or not Path(dev_label_file).exists() or not Path(test_file).exists():
            logging.warning('{} or {} or {} is not existed.'.format(str(dev_file), str(dev_label_file), str(test_file)))
            continue
        report_dict = _get_dev_report(dev_file, dev_label_file)
        final_res = _calc_score(test_file, report_dict, final_res)
    final_y_pred = _weight_sum(final_res)
    _write_vote_result(output_file, final_y_pred)
    return final_y_pred

if __name__ == "__main__":
    from task2.configuration import config
    val_label_files = [config.validate_file['Chinese']]
    val_preds_files = ['a.tsv']
    test_files = ['val_micro@F1=0.6998722860791327.tsv']
    output_file = './zh.pred.conll'
    vote(val_label_files, val_preds_files, test_files, output_file)
    for item1, item2 in zip(read_conll_item_from_file(output_file, just_label=True), read_conll_item_from_file(test_files[0], just_label=True)):
        assert len(item1.labels) == len(item2.labels)
        for l1, l2 in zip(item1.labels, item2.labels):
            assert l1 == l2
