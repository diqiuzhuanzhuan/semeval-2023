# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import logging
import os
import pathlib
import colorlog
from colorlog import ColoredFormatter


root_path = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = root_path/'data'
log_path = root_path/'logs'
output_path = root_path/'output'

wikigaz_file = data_path/'wiki_def'/'wikigaz.tsv.zip'


train_data_path = data_path/'training_data'

train_file = {
    'English': data_path/'training_data'/'EN-English'/'en_train.conll',
 	'Spanish': data_path/'training_data'/'ES-Spanish'/'es_train.conll',
    'Hindi': data_path/'training_data'/'HI-Hindi'/'hi_train.conll',
	'Bangla': data_path/'training_data'/'BN-Bangla'/'bn_train.conll',
	'Chinese': data_path/'training_data'/'ZH-Chinese'/'zh_train.conll',
	'Swedish': data_path/'training_data'/'FA-Farsi'/'fa_train.conll',
	'Farsi': data_path/'training_data'/'FA-Farsi'/'fa_train.conll',
	'French': data_path/'training_data'/'FA-Farsi'/'fa_train.conll',
	'Italian': data_path/'training_data'/'TR-Turkish'/'tr_train.conll',
	'Portugese': data_path/'training_data'/'TR-Turkish'/'tr_train.conll',
	'Ukranian': data_path/'training_data'/'RU-Russian'/'ru_train.conll',
	'German': data_path/'training_data'/'DE-German'/'de_train.conll',
}

validate_file = {
    'English': data_path/'training_data'/'EN-English'/'en_dev.conll',
 	'Spanish': data_path/'training_data'/'ES-Spanish'/'es_dev.conll',
    'Hindi': data_path/'training_data'/'HI-Hindi'/'hi_dev.conll',
	'Bangla': data_path/'training_data'/'BN-Bangla'/'bn_dev.conll',
	'Chinese': data_path/'training_data'/'ZH-Chinese'/'zh_dev.conll',
	'Swedish': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'Farsi': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'French': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'Italian': data_path/'training_data'/'TR-Turkish'/'tr_dev.conll',
	'Portugese': data_path/'training_data'/'TR-Turkish'/'tr_dev.conll',
	'Ukranian': data_path/'training_data'/'RU-Russian'/'ru_dev.conll',
	'German': data_path/'training_data'/'DE-German'/'de_dev.conll',
}

test_file = {
    'English': data_path/'training_data'/'EN-English'/'en_dev.conll',
 	'Spanish': data_path/'training_data'/'ES-Spanish'/'es_dev.conll',
    'Hindi': data_path/'training_data'/'HI-Hindi'/'hi_dev.conll',
	'Bangla': data_path/'training_data'/'BN-Bangla'/'bn_dev.conll',
	'Chinese': data_path/'training_data'/'ZH-Chinese'/'zh_dev.conll',
	'Swedish': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'Farsi': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'French': data_path/'training_data'/'FA-Farsi'/'fa_dev.conll',
	'Italian': data_path/'training_data'/'TR-Turkish'/'tr_dev.conll',
	'Portugese': data_path/'training_data'/'TR-Turkish'/'tr_dev.conll',
	'Ukranian': data_path/'training_data'/'RU-Russian'/'ru_dev.conll',
	'German': data_path/'training_data'/'DE-German'/'de_dev.conll',
}

performance_log = log_path/'performance.csv'

###### for log ######
formatter = ColoredFormatter(
	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logging = colorlog.getLogger('example')
logging.addHandler(handler)
logging.setLevel(colorlog.DEBUG)