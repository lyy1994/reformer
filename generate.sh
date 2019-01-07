#! /usr/bin/bash
set -e

######## hardware (default) ########
# device (-1 for cpu)
device=1

######## dataset (default) ########
# language: zh-en or en-zh
s=de
t=en

# problems
problems=iwslt14

# used for specific data file
data_dir=../data/data-bin

# datatype=SPLIT, e.g. train, valid, test
datatype=test

######## parameters (default) ########
batch_size=16
beam_size=5
lenpen=1
# if use ensemble, set it
ensemble=
# dynamic options, e.g. change some default settings, '--quiet'
other_options="--quiet --remove-bpe"

######## models (default) ########
# must exist
tag=lwcm_small_deep_rich_fat_normb_embed
# used for specific model file
model_file=checkpoint_best.pt

######## evaluation (default) ########
# evaluation or just decoding
is_eval=0
# now available for multi-bleu.perl only
eval_tool=../utils/scripts/multi-bleu.perl

######## args (reset default) ########
# use args to reset default settings
while getopts a:b:c opt; do
  case $opt in
    a)
      echo -e "\033[33mreset datatype from $datatype to $OPTARG by -a\033[0m"
      datatype=$OPTARG
      ;;
    b)
      echo -e "\033[33mreset model_file from $model_file to $OPTARG by -b\033[0m"
      model_file=$OPTARG
      ;;
    c)
      echo -e "\033[33mdisable evaluation by -d\033[0m"
      is_eval=0
      ;;
    \?)
      echo  -e "\033[31mInvalid option: -$OPTARG\033[0m"
      ;;
  esac
done
# if we want to set default value for a variable defined in the while loop, try the following code ("" required for string)
# new_var=${new_var:-default}

######## dataset (no reset) ########
# language: zh-en or en-zh
lang=${s}"-"${t}
# dataset
dataset=${problems}.tokenized.${lang}

######## models (no reset) ########
# used for specific model file
output_dir=../checkpoints/${tag}

######## evaluation (no reset) ########
# used for specific output file
if [ ${is_eval} -eq 1 ]; then
  output_file=${output_dir}/trans.${t}
else
  output_file=
fi
# for single reference evaluation
ref_file_bpe=${data_dir%%/data-bin*}/${dataset}/${datatype}.${t}
ref_file=${output_dir}/${datatype}.${t}


# save generate.sh
cp `pwd`/generate.sh $output_dir

if [ -n "$ensemble" ]; then
  # start ensemble
  model_file=model.pt
  echo -e "\033[34mensemble the last $ensemble checkpoints from $output_dir to $output_dir/$model_file\033[0m"
  if [ -n "$output_dir/$model_file" ]; then
    echo -e "\033[33mWarning: override $output_dir/$model_file\033[0m"
  fi
  PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py \
  --inputs ${output_dir} \
  --num-epoch-checkpoints ${ensemble} \
  --output ${output_dir}/${model_file}
  if [ $? -ne 0 ]; then
    exit -1
  fi
fi

cmd="python3 -u generate.py
$data_dir/$dataset
--gen-subset $datatype
--path $output_dir/$model_file
--batch-size $batch_size
--beam $beam_size
--lenpen $lenpen"
if [ -n "$output_file" ]; then
  cmd=${cmd}" --output-file "${output_file}
fi
if [ -n "$other_options" ]; then
  cmd=${cmd}" "${other_options}
fi

# start decoding
echo -e "\033[34mrun command: "${cmd}"\033[0m"
CUDA_VISIBLE_DEVICES=${device} PYTHONPATH=`pwd` $cmd | tee $output_dir/generate.log

if [ $? -ne 0 ]; then
  exit -1
fi

if [ ${is_eval} -eq 1 ]
then
  # remove bpe
  cp ${ref_file_bpe} ${ref_file}
  sed -i s'/@@ //g' ${ref_file}
  sed -i s'/@@ //g' ${output_file}

  # check eval_tool
  if [ -n "$eval_tool" ]; then
	echo -e "\033[33musing evaluation tool: $eval_tool\033[0m"
  else
	echo -e "\033[31m$eval_tool does not exist!\033[0m"
    exit -1
  fi

  # run eval_tool
  echo -e "\033[34mrun command: perl $eval_tool $ref_file < $output_file\033[0m"
  perl $eval_tool $ref_file < $output_file

  rm $ref_file

fi

