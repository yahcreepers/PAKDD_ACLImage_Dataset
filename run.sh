export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
dataset=$2
logs=$3

python main.py \
--dataset ${dataset} \
--output_dir logs/${logs}/ \
--auto_cl \
--batch_size 16