#! /usr/bin/bash
set -e

devices=${1}
worker_gpus=`echo "$devices" | awk '{n=split($0,arr,",");print n}'`

s=${2}
t=${3}

data_root_dir=${4}
dataset=${5}
subset=${6}

model_dir=${7}
ckpt=${8}

batch_size=${9}
beam_size=${10}
lenpen=${11}

record=${12}

score_reference=${13}
top_k=${14}

log_file=${model_dir}/${ckpt%%.pt}/log_${s}_${t}_${subset}_beam${beam_size}_lenpen${lenpen}
output_file=${model_dir}/${ckpt%%.pt}/hypo_${s}_${t}_${subset}_beam${beam_size}_lenpen${lenpen}
same_file=${model_dir}/${ckpt%%.pt}/same_${s}_${t}_${subset}_beam${beam_size}_lenpen${lenpen}

cmd="python3 -u ../generate.py
$data_root_dir/$dataset
-s $s -t $t
--gen-subset $subset
--path $model_dir/$ckpt
--batch-size $batch_size
--beam $beam_size
--lenpen $lenpen
--output-file $output_file
--same-file $same_file
--quiet
--model-parallelism-world-size $worker_gpus"

if [[ ${score_reference} -eq 1 ]]; then
  cmd=${cmd}" --score-reference --top-k $top_k"
else
  cmd=${cmd}" --remove-bpe"
fi

CUDA_VISIBLE_DEVICES=${devices} PYTHONPATH=`pwd` ${cmd} | tee ${log_file}

if [[ $? -ne 0 ]]; then
  exit -1
fi

bleu=`tail -1 ${log_file} |cut -d " " -f 8| awk '{$a=substr($0,0,length($0)-1);print $a;}'`
bp=`tail -1 ${log_file} |cut -d " " -f 10| awk '{$a=substr($0,5,length($0)-5);print $a;}'`
ratio=`tail -1 ${log_file} |cut -d " " -f 11| awk '{$a=substr($0,7,length($0)-7);print $a;}'`
echo -e "CKPT=${ckpt}\tSet=${subset}\tAlpha=${lenpen}\tBeam=${beam_size}\tBLEU=${bleu}\tBP=${bp}\tRatio=${ratio}" >> ${record}
