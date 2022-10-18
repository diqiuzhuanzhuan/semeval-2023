# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import Any, AnyStr, Union, Optional
from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable
from transformers import AutoTokenizer
import os
from task2.data_man.meta_data import read_conll_item_from_file



class ConllDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    def __getitem__(self, index: Any) -> Any:
        raise NotImplemented('')

    def encode_input(self, item):
        raise NotImplemented('')

    def __len__(self) -> int:
        return len(self.instances)

    def read_data(self, conll_file: Union[AnyStr, os.PathLike]):
        self.instances = read_conll_item_from_file(conll_file)