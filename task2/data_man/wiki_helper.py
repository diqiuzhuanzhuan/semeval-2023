# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import time
from typing import AnyStr
from task2.configuration.config import logging
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
import time
from qwikidata.utils import dump_entities_to_json
from task2.configuration import config
from task2.data_man.meta_data import write_json_gzip
import collections
import pandas as pd
import wikidata_plain_sparql as wikidata
#for t in ['artist', 'politician', 'cleric', 'scientist', 'athlete', 'sportsmanager']:
#    for k in pd.read_csv('{}.tsv'.format(t), header=None).values.flatten().tolist():
#        PERSON_TYPE[k] = t
import sys
from SPARQLWrapper import SPARQLWrapper, JSON, TSV

endpoint_url = "https://query.wikidata.org/sparql"


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
        
P_OCCUPATION = "P106"
PERSON_TYPE = {
    #"Q82955": 'politician',
    #"Q2259532": 'cleric',
    #"Q901": 'scientist',
    "Q483501": 'artist',
    #"Q2066131": 'athlete',
    #'Q838476': 'sportsmanager', #体育总监
    #'Q41583': 'sportsmanager', #教练
}
person_vocab = collections.defaultdict(set)
type_by_id = collections.defaultdict(set)
def query_person(qid):
    query = 'SELECT DISTINCT ?item WHERE {' + \
            '?item p:P106 ?statement0. ' + \
            '?statement0 (ps:P106/(wdt:P279*)) wd:'+qid+'.' + \
            '}' 
    data = get_results(endpoint_url=endpoint_url, query=query)
    ids = [result['item']['value'].replace('http://www.wikidata.org/entity/', '') for result in data["results"]["bindings"]]
    for id in ids:
        type_by_id[id] = PERSON_TYPE[qid]

for k in PERSON_TYPE:
    logging.info(k)
    query_person(k)
#out_fname = "person_entities.json.gz"
#write_json_gzip(out_fname, person_vocab)
logging.info(len(type_by_id))
    

def has_occupation_politician(item: WikidataItem, truthy: bool = True) -> bool:
    """Return True if the Wikidata Item has occupation politician."""
    if truthy:
        claim_group = item.get_truthy_claim_group(P_OCCUPATION)
    else:
        claim_group = item.get_claim_group(P_OCCUPATION)

    occupation_qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]
    if len(occupation_qids) == 0:
        return []
    types = set()
    for k in occupation_qids:
        if k in PERSON_TYPE:
            types.add(PERSON_TYPE[k])
            break
        if len(types):
            break

    if not types:
        types.add('person')
    return types


# create an instance of WikidataJsonDump
wjd_dump_path = '/Users/malong/Downloads/wikidata-20220103-all.json.gz'
wjd = WikidataJsonDump(wjd_dump_path)

# create an iterable of WikidataItem representing politicians
person = []
t1 = time.time()
person_vocab = collections.defaultdict(set)
for ii, entity_dict in enumerate(wjd):
    
    if entity_dict["type"] == "item":
        entity = WikidataItem(entity_dict)
        types = has_occupation_politician(entity)
        if types:
            labels = entity._entity_dict.get('labels')
            for lan in config.ios_639_1_code:
                value = labels.get(lan, dict()).get('value', '')
                person_vocab[value].update(set(types)) if value else None
            
    if ii % 10000 == 0:
        t2 = time.time()
        dt = t2 - t1
        print(
            "found {} person among {} entities [entities/s: {:.2f}]".format(
                len(person_vocab), ii, ii / dt
            )
        )

print(person_vocab)
out_fname = "person_entities.json.gz"
import json
for k in person_vocab:
    person_vocab[k] = sorted(list(person_vocab[k]))

with open(out_fname, 'w') as f:
    json.dump(person_vocab, f)
#dump_entities_to_json(person_vocab, out_fname)


# write the iterable of WikidataItem to disk as JSON
#out_fname = "filtered_entities.json"
#dump_entities_to_json(person, out_fname)
#wjd_filtered = WikidataJsonDump(out_fname)

# load filtered entities and create instances of WikidataItem
#for ii, entity_dict in enumerate(wjd_filtered):
#    item = WikidataItem(entity_dict)
#    print(item)