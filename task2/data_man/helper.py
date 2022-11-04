# -*- coding: utf-8 -*- 
# # author: Feynman 
# # email: diqiuzhuanzhuan@gmail.com

import os, json, aiohttp, asyncio
from typing import AnyStr, Dict, Iterable
import urllib.request
import urllib.parse
from task2.data_man.meta_data import get_wiki_title_knowledge, get_wiki_title_google_type, write_wiki_title_google_type
from task2.configuration import config
from task2.configuration.config import logging
from tqdm import tqdm
import grequests

api_key = 'AIzaSyAYG8RO5R50nae4S9TsYAAckCRaR0yHQZI'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

def build_params(query):
    params = {
        'query': query,
        'limit': 1,
        'indent': True,
        'key': api_key,
    }
    return params


def build_request(querys):
    gets = []
    for query in querys:
        params = build_params(query=query)
        url = service_url + '?' + urllib.parse.urlencode(params)
        gets.append(grequests.get(url))
    return gets

def build_response(query, response):
    ans = dict()
    if not response or response.status_code != 200:
        logging.error('visit failed for {}'.format(query))
        return ans

    data = response.json()
    if 'itemListElement' not in data or len(data['itemListElement']) == 0:
        logging.warning('{} is not in google.'.format(query))
        ans[query] = dict()
    else:
        significant_part = data['itemListElement'][0]['result']
        ans[query] = {
            'type': significant_part['@type'] if '@type' in significant_part else None,
            'description': significant_part['description'] if 'description' in significant_part else None
            }
    return ans
    
def get_google_knowledge(querys: Iterable[AnyStr], ans: Dict):
    batch_querys = []
    for query in tqdm(querys, total=len(querys)):
        if query in ans:
            continue
        if len(batch_querys) < 50:
            batch_querys.append(query)
            continue
        task_list = build_request(batch_querys)
        res_list = grequests.map(task_list)
        ans_list = [build_response(query, response) for query, response in zip(batch_querys, res_list)]
        for item in ans_list:
            ans.update(item)
        batch_querys.clear()

    if batch_querys:
        task_list = build_request(batch_querys)
        res_list = grequests.map(task_list)
        ans_list = [build_response(query, response) for query, response in zip(batch_querys, res_list)]
        for item in ans_list:
            ans.update(item)
    return ans

def main(lang: AnyStr='Chinese'):
    entity_vocab = get_wiki_title_knowledge(config.wiki_title_file[lang])
    querys = entity_vocab.keys()
    ans = get_wiki_title_google_type(config.wiki_title_with_google_type_file[lang])
    ans = get_google_knowledge(querys=querys, ans=ans)
    write_wiki_title_google_type(file=config.wiki_title_with_google_type_file[lang], wiki_knowledge=ans)
#    await session.close()

if __name__ == "__main__":
    main('Chinese')
