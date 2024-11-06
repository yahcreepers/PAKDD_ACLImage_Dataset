export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
dataset=$2
round=$3
logs=$4

python main.py \
--dataset ${dataset} \
--output_dir logs/${logs}/ \
--batch_size 24 \
--num_rounds ${round} \
--long_label \
--auto_cl \
