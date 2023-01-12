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
kfold_data_path = data_path/'kfold'
log_path = root_path/'logs'
output_path = root_path/'output'

wikigaz_file = data_path/'wiki_def'/'wikigaz.tsv.zip'
wiki_title_file = {
	'Chinese': data_path/'wiki_def'/'zhwiki-latest-all-titles.gz'
}

wiki_title_with_google_type_file = {
    'Chinese': data_path/'wiki_def'/'chinese_wiki_google_file.gz'
}

wiki_entity_data = {
    'person': data_path/'wiki_def'/'person_entities.json.gz',
    'group': data_path/'wiki_def'/'group_entities.json.gz',
    'medicine': data_path/'wiki_def'/'medicine_entities.json.gz',
	'location': data_path/'wiki_def'/'location_entities.json.gz',
	'product': data_path/'wiki_def'/'product_entities.json.gz',
	'creative work': data_path/'wiki_def'/'creative_work_entities.json.gz'
}

wiki_data = {
    'English': data_path/'wiki_def'/'person_entities.json.gz',
 	'Spanish': data_path/'wiki_def'/'person_entities.json.gz',
    'Hindi': data_path/'wiki_def'/'person_entities.json.gz',
	'Bangla': data_path/'wiki_def'/'person_entities.json.gz',
	'Chinese': data_path/'wiki_def'/'person_entities.json.gz',
	'Swedish': data_path/'wiki_def'/'person_entities.json.gz',
	'Farsi': data_path/'wiki_def'/'person_entities.json.gz',
	'French': data_path/'wiki_def'/'person_entities.json.gz',
	'Italian': data_path/'wiki_def'/'person_entities.json.gz',
	'Portugese': data_path/'wiki_def'/'person_entities.json.gz',
	'Ukranian': data_path/'wiki_def'/'person_entities.json.gz',
	'German': data_path/'wiki_def'/'person_entities.json.gz'
}

ios_639_1_code = set([
	'en', 'es', 'hi', 'bn', 'zh', 'sv', 'fa', 'fr', 'it', 'pt', 'uk', 'de'
])

code_by_lang = {
    'English': 'en',
    'Spanish': 'es',
	'Hindi': 'hi',
	'Bangla': 'bn',
	'Chinese': 'zh',
	'Swedish': 'sv',
	'Farsi': 'fa',
	'French': 'fr',
	'Italian': 'it',
	'Portugese': 'pt',
	'Ukranian': 'uk',
	'German': 'de'
}

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
    'All': train_data_path/'all-train.conll',
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
	'All': validate_data_path/'all-dev.conll',
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
	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s %(filename)s line:%(lineno)d %(message)s",
	datefmt='%Y-%m-%d %H:%M:%S',
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