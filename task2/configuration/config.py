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


train_data_path = data_path/'train_dev'
validate_data_path = data_path/'train_dev'
test_data_path = data_path/'train_dev'

train_file = {
    'English': train_data_path/'en-train.conll',
 	'Spanish': train_data_path/'es-train.conll',
    'Hindi': train_data_path/'hi-train.conll',
	'Bangla': train_data_path/'bn-train.conll',
	'Chinese': train_data_path/'zh-train.conll',
	'Swedish': train_data_path/'sv-train.conll',
	'Farsi': train_data_path/'fa-train.conll',
	'French': train_data_path/'fr-train.conll',
	'Italian': train_data_path/'it-train.conll',
	'Portugese': train_data_path/'pt-train.conll',
	'Ukranian': train_data_path/'uk-train.conll',
	'German': train_data_path/'de-train.conll',
}

validate_file = {
    'English': validate_data_path/'en-dev.conll',
 	'Spanish': validate_data_path/'es-dev.conll',
    'Hindi': validate_data_path/'hi-dev.conll',
	'Bangla': validate_data_path/'bn-dev.conll',
	'Chinese': validate_data_path/'zh-dev.conll',
	'Swedish': validate_data_path/'sv-dev.conll',
	'Farsi': validate_data_path/'fa-dev.conll',
	'French': validate_data_path/'fr-dev.conll',
	'Italian': validate_data_path/'it-dev.conll',
	'Portugese': validate_data_path/'pt-dev.conll',
	'Ukranian': validate_data_path/'uk-dev.conll',
	'German': validate_data_path/'de-dev.conll',
}

test_file = {
	'English': validate_data_path/'en-dev.conll',
 	'Spanish': validate_data_path/'es-dev.conll',
    'Hindi': validate_data_path/'hi-dev.conll',
	'Bangla': validate_data_path/'bn-dev.conll',
	'Chinese': validate_data_path/'zh-dev.conll',
	'Swedish': validate_data_path/'sv-dev.conll',
	'Farsi': validate_data_path/'fa-dev.conll',
	'French': validate_data_path/'fr-dev.conll',
	'Italian': validate_data_path/'it-dev.conll',
	'Portugese': validate_data_path/'pt-dev.conll',
	'Ukranian': validate_data_path/'uk-dev.conll',
	'German': validate_data_path/'de-dev.conll',
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