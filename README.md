# Auto Labeling

## Installation

`pip install -r requirements.txt`

## Run the code:

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

`./run.sh <device number> <dataset name> <output_dir>`

## Results:
https://docs.google.com/spreadsheets/d/1tf-2_AfH_D_ZFmvpxGMCRhlTHaehva4D1EUrhW7efu0/edit?usp=sharing

## References:
https://huggingface.co/docs/transformers/model_doc/llava_next