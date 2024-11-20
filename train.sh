export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=$3
model=$4
dataset=$5
label_path=$6
lr=$7
batch_size=$8
seed=$9

output_dir="train_logs/${strategy}/${dataset}/${strategy}-${tp}-${model}-${dataset}-${lr}-${seed}"
# output_dir="train_logs/test/"
python train.py \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --epoch 300 \
    --valid_type Accuracy \
    --output_dir ${output_dir} \
    --label_path ${label_path} \
    --long_label \
    --do_transform \
    --seed ${seed} \
    
