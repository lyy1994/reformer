#! /usr/bin/bash
set -e

tool_path=${1}

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

log_file=${model_dir}/${ckpt%%.pt}/log_${s}_${t}_${subset}_beam${beam_size}_lenpen${lenpen}
output_file=${model_dir}/${ckpt%%.pt}/hypo_${s}_${t}_${subset}_beam${beam_size}_lenpen${lenpen}

# for multiple references evaluation
ref_file_bpe=`find ${data_root_dir%%/data-bin*}/${dataset} -name ${subset}[0-9]*.${t}`
if [[ -z "$ref_file_bpe" ]]; then
  ref_file_bpe=${data_root_dir%%/data-bin*}/${dataset}/${subset}.${t}
fi
ref_file=${model_dir}/${ckpt%%.pt}/${subset}*.${t}

echo -e "| Detected References:\n$ref_file_bpe" | tee -a -i ${log_file}

# remove bpe
cp ${ref_file_bpe} ${model_dir}/${ckpt%%.pt}
sed -i s'/@@ //g' ${ref_file}
sed -i s'/@@ //g' ${output_file}

# run evaluation script
cmd="perl ${tool_path}/multi-bleu.perl -lc $ref_file < $output_file"
echo -e "\033[34m| $cmd\033[0m" | tee -a -i ${log_file}
eval ${cmd} | tee -a -i ${log_file}

if [[ $? -ne 0 ]]; then
  exit -1
fi

rm ${ref_file}

bleu=`tail -1 ${log_file} | cut -d " " -f 3 | awk '{$a=substr($0,0,length($0)-1);print $a;}'`
bp=`tail -1 ${log_file} | cut -d " " -f 5 | awk '{$a=substr($0,5,length($0)-5);print $a;}'`
ratio=`tail -1 ${log_file} | cut -d " " -f 6 | awk '{$a=substr($0,7,length($0)-7);print $a;}'`
echo -e "CKPT=${ckpt}\tSet=${subset}\tAlpha=${lenpen}\tBeam=${beam_size}\tBLEU=${bleu}\tBP=${bp}\tRatio=${ratio}" >> ${record}

