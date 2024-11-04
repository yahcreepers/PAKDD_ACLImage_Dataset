# Auto Labeling

## Installation

```
git clone https://github.com/yahcreepers/auto_labeling.git
cd auto_labeling
pip install -r requirements.txt
cd ..
git clone https://github.com/ntucllab/libcll.git
cd libcll
pip install -e .
```

## For Auto-Labeling

```
python main.py \
--model llava \
--dataset cifar10 \
--batch_size 8 \
--seed 1126 \
--num_rounds 3 \
--output_dir logs/ \
--auto_cl \ # collecting complementary labels or ordinary labels
--long_label # using whole label names instead of only one word abbreviation
```

or

```
./run.sh <device number> <dataset name> <epoch> <output_dir>
```

## For Training

```
./train.sh <device number> <strategy> <type> <model> <dataset> <label path>
```

## Results
https://docs.google.com/spreadsheets/d/1tf-2_AfH_D_ZFmvpxGMCRhlTHaehva4D1EUrhW7efu0/edit?usp=sharing

## References
https://huggingface.co/docs/transformers/model_doc/llava_next