#!/bin/bash

SHELL_FOLDER=$(dirname "$0")
cd $SHELL_FOLDER
PYTHONPATH=$(dirname $(dirname $(pwd)}))
export PYTHONPATH=$PYTHONPATH
cd ".."
echo "当前工作路径: $(pwd)"


run()
{
    for encoder_model in  'bert-base-multilingual-uncased' 'xlm-roberta-base' 'xlm-roberta-large' 'bert-base-chinese' 'hfl/chinese-roberta-wwm-ext' 'hfl/chinese-bert-wwm-ext'
    do
        for max_epochs in 25
        do
            for batch_size in 16
            do
                for model_type in 'baseline_crf_model' 'baseline_ner_model'
                do
                    for dataset_type in 'dictionary_fused_dataset' 'baseline_dataset'
                    do
                        for data_module_type in 'baseline_data_module'
                        do
                            python apps/main.py \
                            --encoder_model $encoder_model \
                            --max_epochs $max_epochs \
                            --batch_size $batch_size \
                            --model_type $model_type \
                            --dataset_type $dataset_type \
                            --data_module_type $data_module_type \
                            --gpus -1 \
                            --lang $1

                        done
                    done
                done            
            done
        done 
    done
}

#for lang in 'English' 'Spanish' 'Hindi' 'Bangla' 'Chinese' 'Swedish' 'Farsi' 'French' 'Italian' 'Portugese' 'Ukranian' 'German'
# test over Chinese dataset
for lang in 'Chinese'
do
    echo "run for $lang"
    run $lang
    echo "finish for $lang"
done