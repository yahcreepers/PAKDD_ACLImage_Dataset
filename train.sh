export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=$3
model=$4
dataset=$5
label_path=$6
lr=$7

output_dir="train_logs/${strategy}/${dataset}/${strategy}-${tp}-${model}-${dataset}"
python train.py \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr ${lr} \
    --batch_size 256 \
    --valid_type Accuracy \
    --output_dir ${output_dir} \
    --label_path ${label_path} \
    --long_label \
    --do_transform \
    
