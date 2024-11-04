cuda=$1

# strategies=("SCL" "URE" "CPE")
strategies=("SCL" "CPE")
# datasets=("min10" "cifar10" "min20" "cifar20")
datasets=("min10" "cifar10")

for strategy in ${strategies[@]}; do
    for dataset in ${datasets[@]}; do
        valid_type="URE"
        if [[ $strategy == "SCL" ]]; then
            types=("NL" "EXP" "FWD")
            types=("NL" "FWD")
        elif [[ $strategy == "URE" ]]; then
            types=("NN" "GA" "TNN" "TGA")
        elif [[ $strategy == "CPE" ]]; then
            types=("I" "F" "T")
            types=("T")
            valid_type="SCEL"
        elif [[ $strategy == "FWD" ]] || [[ $strategy == "DM" ]]; then
            types=("None")
        elif [[ $strategy == "MCL" ]]; then
            types=("MAE" "EXP" "LOG")
        fi
        model="ResNet34"
        valid_type="Accuracy"
        for t in ${types[@]}; do
			echo "./train.sh ${cuda} ${strategy} ${t} ${model} ${dataset} logs/${dataset}_origin"
            ./train.sh ${cuda} ${strategy} ${t} ${model} ${dataset} logs/${dataset}_origin
        done
    done
done
