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
    
    cl_labels = []
    with tqdm(total=args.num_rounds * len(train_dataloader)) as pbar:
        for round in range(args.num_rounds):
            total_answers = []
            for step, data in enumerate(train_dataloader):
                # if step >= 5:
                #     break
                images, labels = data
                prompts = [model.create_cll_prompt(train_set.label_map) for _ in range(len(images))]
                answers = model.predict(images, prompts)
                answers = [train_set.class_to_idx[answer] for answer in answers]
                # answers = list(np.random.choice(range(10), len(images)))
                # answers = list(int(np.random.choice(list(set(range(10)) - {labels[i].item()}), 1)) for i in range(len(images)))
                total_answers.extend(answers)
                pbar.update(1)
            cl_labels.append(total_answers)
    cl_labels = torch.tensor(cl_labels).t().long()
    noise = (train_set.targets.expand_as(cl_labels) == cl_labels).float().mean()
    Q = torch.zeros((train_set.num_classes, train_set.num_classes))
    for target, cl in zip(train_set.targets, cl_labels):
        Q[target.long()] += torch.bincount(cl, minlength=train_set.num_classes)
    Q = Q / Q.sum(dim=1).view(-1, 1)
    print("Noise", noise.item())
    print("Transition Matrix", Q)
    with open(os.path.join(args.output_dir, "cl_labels.csv"), "w") as f:
        for cl in cl_labels:
            cl = ",".join(train_set.label_map[label] for label in cl)
            f.write(f"{cl}\n")
    with open(os.path.join(args.output_dir, "logs.csv"), "w") as f:
        print("Noise", noise.item(), file=f)
        print("Transition Matrix", Q, file=f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument(
        "--model_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf"
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--output_dir", type=str, default="logs/")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)