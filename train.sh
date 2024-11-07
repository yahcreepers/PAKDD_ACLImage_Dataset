export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=$3
model=ResNet18
dataset=$4
label_path=$5
lr=$6

output_dir="train_logs/${strategy}/${dataset}/${strategy}-${tp}-${model}-${dataset}"
python train.py \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr ${lr} \
    --batch_size 256 \
    --epoch 600 \
    --valid_type Accuracy \
    --output_dir ${output_dir} \
    --label_path ${label_path} \
    --long_label \
    --do_transform \
    
