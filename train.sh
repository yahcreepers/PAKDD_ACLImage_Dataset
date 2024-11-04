export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=$3
model=$4
dataset=$5
label_path=$6

output_dir="/tmp2/yahcreeper/auto_labeling/train_logs/${strategy}/${dataset}/${strategy}-${tp}-${model}-${dataset}"
python train.py \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr 1e-4 \
    --batch_size 256 \
    --valid_type Accuracy \
    --output_dir ${output_dir} \
    --label_path ${label_path} \
    --long_label \
    --do_transform \
    