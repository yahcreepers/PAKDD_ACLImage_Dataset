cuda=$1
strategy=$2
type=$3
dataset_set=$4

# strategies=("SCL" "URE" "CPE")
# strategies=("URE" "CPE")
strategies=(${strategy})
# strategies=("SCL")
# datasets=("min10" "cifar10" "min20" "cifar20")
# datasets=("min20" "cifar20")
if [[ ${dataset_set} == 0 ]]; then
    datasets=("min10" "cifar10")
elif [[ ${dataset_set} == 1 ]]; then
    datasets=("min20" "cifar20")
fi
seeds=("1207" "9213" "17" "33")
# datasets=("cifar20")
# datasets=("min10")

for strategy in ${strategies[@]}; do
    for dataset in ${datasets[@]}; do
        valid_type="URE"
        if [[ $strategy == "SCL" ]]; then
            types=("NL" "EXP" "FWD")
            types=("NL" "FWD")
        elif [[ $strategy == "URE" ]]; then
            types=("NN" "GA" "TNN" "TGA")
            types=("GA" "TGA")
        elif [[ $strategy == "CPE" ]]; then
            types=("I" "F" "T")
            valid_type="SCEL"
        elif [[ $strategy == "FWD" ]] || [[ $strategy == "DM" ]]; then
            types=("None")
        elif [[ $strategy == "MCL" ]]; then
            types=("MAE" "EXP" "LOG")
        fi
        model="ResNet34"
        valid_type="Accuracy"
        types=(${type})
        for t in ${types[@]}; do
            lrs=($(cat "/work/u8273333/libcll/logs/${strategy}/${strategy}-${t}-multi-hard.txt"))
            echo ${lrs[@]}
            if [[ $dataset == "min10" ]] || [[ $dataset == "clmin10" ]]; then
                lr=${lrs[0]}
                batch_size=256
            elif [[ $dataset == "min20" ]] || [[ $dataset == "clmin20" ]]; then
                lr=${lrs[1]}
                batch_size=256
            elif [[ $dataset == "cifar10" ]] || [[ $dataset == "clcifar10" ]]; then
                lr=${lrs[2]}
                batch_size=256
            elif [[ $dataset == "cifar20" ]] || [[ $dataset == "clcifar20" ]]; then
                lr=${lrs[3]}
                batch_size=256
            fi
            for seed in ${seeds[@]}; do
                echo "./train.sh ${cuda} ${strategy} ${t} ${model} ${dataset} logs/${dataset}_origin ${lr} ${batch_size} ${seed}"
                # ./train.sh ${cuda} ${strategy} ${t} ${model} ${dataset} logs/${dataset}_origin ${lr} ${batch_size} ${seed}
            done
        done
    done
done
