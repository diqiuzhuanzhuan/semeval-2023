#!/bin/bash


wget --continue https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
cat latest-all.json.gz | gzip -d | wikibase-dump-filter --claim P31:Q5 > human.json
cat latest-all.json.gz | gzip -d | wikibase-dump-filter --claim P31:Q43229 > org.json
cat latest-all.json.gz | gzip -d | wikibase-dump-filter --claim P31:Q28453457 > location.json

