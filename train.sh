#! /usr/bin/bash
set -e 

######## hardware ########
# devices
devices=3,4,5,6,7

######## dataset ########
# language: zh-en or en-zh
s=de
t=en
lang=${s}"-"${t}
# dataset
dataset=iwslt14.tokenized.${lang}

######## parameters ########
# which model
model=reformer
# which hparams 
param=${model}"_iwslt_"${s}"_"${t}
# defualt is 103k. About 10 epochs for 700w CWMT
max_update=30000
# dynamic hparams, e.g. change the batch size without the register in code, other_hparams='batch_size=2048'
other_hparams=

######## required ########
# tag is the name of your experiments
tag=reformer_e256_m256_l10_avg_attn_dropb_normb_decb_encffn_dropout01_attndrop01_decay0_opt



# automatically set worker_gpu according to $devices
worker_gpu=`echo "$devices" | awk '{split($0,arr,",");print length(arr)}'`
# dir of training data
data_dir=../data/data-bin
# dir of models
output_dir=../checkpoints/${tag}

if [ ! -d "$output_dir" ]; then
  mkdir -p ${output_dir}
else
  echo -e "\033[31m$output_dir exists!\033[0m"
  exit -1
fi
# save train.sh
cp `pwd`/train.sh $output_dir

if [ ! -d "$data_dir/$dataset" ]; then
  # start preprocessing
  echo -e "\033[34mpreprocess from ${data_dir%%/data-bin*}/$dataset to $data_dir/$dataset\033[0m"
  python3 -u preprocess.py \
  --source-lang ${s} \
  --target-lang ${t} \
  --trainpref ${data_dir%%/data-bin*}/${dataset}/train \
  --validpref ${data_dir%%/data-bin*}/${dataset}/valid \
  --testpref ${data_dir%%/data-bin*}/${dataset}/test \
  --destdir ${data_dir}/${dataset}
fi

adam_betas="'(0.9, 0.997)'"

cmd="python3 -u train.py
$data_dir/$dataset
-a $param
-s $s
-t $t

--encoder-embed-dim 256
--decoder-embed-dim 256
--decoder-ffn-embed-dim 1024
--decoder-attention-heads 4
--decoder-input-layer add
--decoder-output-layer attn
--flow sequential
--scaling mean
--decoder-normalize-before
--decoder-layers 10
--decoder-sublayer-before
--encoder-ffn
--dropout-before
--attention-dropout 0.1
--relu-dropout 0
--dropout 0.1

--distributed-world-size 1
--model-parallelism
--model-parallelism-debug

--no-progress-bar
--log-interval 100

--max-update $max_update
--max-tokens 250
--update-freq 16

--criterion label_smoothed_cross_entropy
--label-smoothing 0.1
--weight-decay 0

--lr-scheduler inverse_sqrt
--warmup-updates 8000
--warmup-init-lr 1e-07
--min-lr 1e-09
--lr 0.001

--save-dir $output_dir

--optimizer adam"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ -n "$other_hparams" ]; then
  cmd=${cmd}" "${other_hparams}
fi

echo -e "\033[34mrun command: "${cmd}"\033[0m"
# start training
cmd="CUDA_VISIBLE_DEVICES=$devices nohup "${cmd}" > $output_dir/train.log 2>&1 &"
eval $cmd

# monitor training log
tail -f ${output_dir}/train.log
