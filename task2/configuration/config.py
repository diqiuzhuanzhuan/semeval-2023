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

train_data_path = data_path/'training_data'

train_file = {
}


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