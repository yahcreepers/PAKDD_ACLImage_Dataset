import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser
import random
from datasets import prepare_dataset
from models import prepare_model
from torch.utils.data import DataLoader
from tqdm import tqdm

def collate_fn(batch):
    images = []
    labels = []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    return images, torch.tensor(labels)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(args):
    set_seed(args.seed)
    train_set, test_set = prepare_dataset(args)
    model = prepare_model(args)
    set_seed(args.seed)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)
    
    total_answers = []
    total_labels = []
    for step, data in enumerate(tqdm(train_dataloader)):
        # if step >= 5:
        #     break
        images, labels = data
        prompts = [model.create_cll_prompt(train_set.label_map) for _ in range(len(images))]
        answers = model.predict(images, prompts)
        answers = torch.tensor([train_set.label2id[answer] for answer in answers])
        total_answers.append(answers)
        total_labels.append(labels)
    
    total_answers = torch.cat(total_answers, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    noise = (total_labels == total_answers).float().mean()
    Q = torch.zeros((train_set.num_classes, train_set.num_classes))
    for i, label in enumerate(total_labels):
        Q[label.long()][total_answers[i].long()] += 1
    Q = Q / Q.sum(dim=1).view(-1, 1)
    print("Noise", noise.item())
    print("Transition Matrix", Q)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument(
        "--model_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf"
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1126)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)