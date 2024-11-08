export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
dataset=$2
round=$3
log_dir="logs/cl_${dataset}_origin_${round}_dif"

python main.py \
--dataset ${dataset} \
--output_dir ${log_dir} \
--batch_size 16 \
--num_rounds ${round} \
--long_label \
--auto_cl \
