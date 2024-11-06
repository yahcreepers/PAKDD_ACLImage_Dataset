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
import csv

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
    
    cl_set = {}
    if not args.auto_cl:
        cl_set = {os.path.splitext(os.path.basename(file_name))[0]: list(train_set.class_to_idx.keys()) for file_name in train_set.names}
    elif args.dataset == "min10" or args.dataset == "min20":
        with open(f"input_cll_tinyimgnet{args.dataset[-2:]}_uniformly.csv", "r") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.split(",")
                for ind in range(10):
                    if args.long_label:
                        cls = [l.strip().lower() for l in line[ind * 5 + 1:(ind + 1) * 5]]
                    else:
                        cls = [l.split(" ")[-1].strip().lower() for l in line[ind * 5 + 1:(ind + 1) * 5]]
                    cl_set[os.path.splitext(os.path.basename(line[ind * 5]))[0]] = cls
    else:
        for file_name in train_set.names:
            cl_set[os.path.splitext(os.path.basename(file_name))[0]] = list(np.random.choice(train_set.label_map, 4, replace=False))
    
    auto_labels = []
    errors = 0
    total_steps = 0
    with tqdm(total=args.num_rounds * len(train_dataloader)) as pbar:
        for round in range(args.num_rounds):
            total_answers = []
            for step, data in enumerate(train_dataloader):
                images, labels = data
                if args.auto_cl:
                    prompts = [model.create_cl_prompt(train_set.label_map, cl_set[os.path.splitext(train_set.names[step * args.batch_size + _])[0]]) for _ in range(len(images))]
                else:
                    prompts = [model.create_ol_prompt(train_set.label_map) for _ in range(len(images))]
                answers = model.predict(images, prompts)
                for i, (image, prompt, answer) in enumerate(zip(images, prompts, answers)):
                    flag = 0
                    while not flag:
                        label = ""
                        candid_set = cl_set[os.path.splitext(train_set.names[step * args.batch_size + i])[0]]
                        for candid in candid_set:
                            if answer in candid or (answer[-1] == "s" and answer[:-1] in candid) or answer.replace(" ", "") in candid.replace(" ", "") or set(answer.split(" ")) == set(candid.split(" ")):
                                label = train_set.class_to_idx[candid]
                                flag += 1
                        if flag != 1:
                            print(f"Error: Step {step} {prompt} {flag} {candid_set}, {answer}")
                            if args.auto_cl:
                                prompt = model.create_cl_prompt(train_set.label_map, candid_set)
                            else:
                                prompt = model.create_ol_prompt(train_set.label_map)
                            errors += 1
                            answer = model.predict(image, prompt)[0]
                            flag = 0
                        else:
                            total_answers.append(label)
                        total_steps += 1
                # answers = [train_set.class_to_idx[answer] for answer in answers]
                # answers = list(np.random.choice(range(10), len(images)))
                # answers = list(int(np.random.choice(list(set(range(10)) - {labels[i].item()}), 1)) for i in range(len(images)))
                # total_answers.extend(answers)
                pbar.update(1)
            auto_labels.append(total_answers)
    auto_labels = torch.tensor(auto_labels).t().long()
    with open(os.path.join(args.output_dir, "auto_labels.csv"), "w") as f:
        for name, cl in zip(train_set.names, auto_labels):
            cl = ",".join([name] + [train_set.label_map[label] for label in cl])
            f.write(f"{cl}\n")
    noise = (train_set.targets.view(-1, 1).expand_as(auto_labels) == auto_labels).float().mean()
    if not args.auto_cl:
        noise = 1 - noise
        if args.num_rounds > 1:
            vote_noise = (train_set.targets != auto_labels.mode(-1)[0]).float().mean()
    Q = torch.zeros((train_set.num_classes, train_set.num_classes))
    for target, cl in zip(train_set.targets, auto_labels):
        Q[target.long()] += torch.bincount(cl, minlength=train_set.num_classes)
    Q = Q / Q.sum(dim=1).view(-1, 1)
    print("Noise", noise.item())
    if args.num_rounds > 1:
        print("Vote Noise", vote_noise.item())
    print("Error", errors / total_steps)
    print("Transition Matrix", Q)
    with open(os.path.join(args.output_dir, "logs.csv"), "w") as f:
        print("Noise", noise.item(), file=f)
        if args.num_rounds > 1:
            print("Vote Noise", vote_noise.item(), file=f)
        print("Error", errors / total_steps, file=f)
        print("Transition Matrix", Q, file=f)
    with open(os.path.join(args.output_dir, "label_set.csv"), "w") as f:
        for file_name in train_set.names:
            cl = ",".join([file_name] + cl_set[os.path.splitext(os.path.basename(file_name))[0]])
            f.write(f"{cl}\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_next")
    parser.add_argument(
        "--model_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf"
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--output_dir", type=str, default="logs/")
    parser.add_argument("--auto_cl", action="store_true")
    parser.add_argument("--long_label", action="store_true")
    parser.add_argument("--do_transform", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
