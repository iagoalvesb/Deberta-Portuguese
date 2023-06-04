#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/RTD/

max_seq_length=512
num_training_steps=100
num_train_epochs=1
learning_rate=1e-4
train_batch_size=128
warmup=0.01
accumulative_update=1


data_dir=$cache_dir/dataset

function setup_wiki_data(){
	task=$1

	if [[ ! -e  $data_dir/valid.txt ]]; then
		mkdir -p $data_dir
       		python ./prepare_data.py -i ./dataset/train.txt -o $data_dir/train.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i ./dataset/valid.txt -o $data_dir/valid.txt --max_seq_length $max_seq_length
		# python ./prepare_data.py -i ./dataset/test.txt -o $data_dir/test.txt --max_seq_length $max_seq_length
	fi
}

setup_wiki_data

Task=RTD

init=$1
tag=$init
case ${init,,} in
	deberta-v3-base)
	parameters=" --num_train_epochs $num_train_epochs \
	--model_config rtd_base.json \
	--warmup $warmup \
	--learning_rate $learning_rate \
	--train_batch_size $train_batch_size \
    	--accumulative_update $accumulative_update \
	--decoupled_training True \
	--fp16 True "
		;;
esac

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps $num_training_steps  \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path tokenizer/spm.model \
	--vocab_type spm \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
