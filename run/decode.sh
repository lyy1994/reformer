#! /usr/bin/bash
set -e

devices=0,1,2,3
subsets=(valid test)

s=de
t=en
dataset=iwslt14.tokenized.de-en

tag=iwslt_reformer
ckpt=checkpoint_best.pt

batch_size=4
beam_size=5
lenpen=1
ensemble=

is_eval=1
tool_path=../../toolkit

data_root_dir=../../data/data-bin
model_dir=../../checkpoints/torch-1.1.0/${tag}

if [[ -n "$ensemble" ]]; then
  # start ensemble
  ckpt=checkpoint_ensemble_last${ensemble}.pt
  echo -e "\033[34m| ensemble the last $ensemble checkpoints from ${model_dir} to ${model_dir}/${ckpt}\033[0m"
  if [[ -f "${model_dir}/${ckpt}" ]]; then
    echo -e "\033[33m| Warning: override ${model_dir}/${ckpt}\033[0m"
  fi
  PYTHONPATH=`pwd` python3 ../scripts/average_checkpoints.py \
  --inputs ${model_dir} \
  --num-epoch-checkpoints ${ensemble} \
  --output ${model_dir}/${ckpt}
  if [[ $? -ne 0 ]]; then
    exit -1
  fi
fi

if [[ ! -d "${model_dir}/${ckpt%%.pt}" ]]; then
  mkdir -p ${model_dir}/${ckpt%%.pt}
fi

# save this script
cp `pwd`/${BASH_SOURCE[0]} ${model_dir}/${ckpt%%.pt}

translate_record=`mktemp -t temp.translate.XXXXXX`
eval_record=`mktemp -t temp.eval.XXXXXX`

echo -e "\033[34m| src=${s} tgt=${t} ckpt=${ckpt} beam=${beam_size} lenpen=${lenpen}\033[0m"
for subset in ${subsets[@]}
do
  echo -e "\033[34m| dev=${devices} subset=${subset}\033[0m"
  sh translate.sh ${devices} ${s} ${t} ${data_root_dir} ${dataset} ${subset} ${model_dir} ${ckpt} ${batch_size} ${beam_size} ${lenpen} ${translate_record}
  wait
  if [[ ${is_eval} -eq 1 ]]; then
    sh eval.sh ${tool_path} ${s} ${t} ${data_root_dir} ${dataset} ${subset} ${model_dir} ${ckpt} ${batch_size} ${beam_size} ${lenpen} ${eval_record}
  fi
done

echo -e "\033[34m| Internal Evaluation:\033[0m"
cat ${translate_record} | sort -u -r

if [[ ${is_eval} -eq 1 ]]; then
  echo -e "\033[34m| External Evaluation:\033[0m"
  cat ${eval_record} | sort -u -r
fi

rm ${translate_record}
rm ${eval_record}

