import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import numpy as np
import copy
import parse


class LLAVA_NEXT:
    def __init__(self, model_path="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
            device_map="auto", 
        )
    
    @classmethod
    def build_model(self, args):
        return self(args.model_path)
    
    def predict(self, image, prompt, options):
        inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=30, pad_token_id=self.processor.tokenizer.eos_token_id)
        answers = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        processed_answers = []
        format_strings = ['({0}) {1}', '({0})']
        for answer, option in zip(answers, options):
            answer = answer.split("[/INST]")[-1].strip().replace(".", "").lower()
            for format_string in format_strings:
                format_answer = parse.parse(format_string, answer)
                if format_answer != None:
                    answer = format_answer[-1]
                    break
            if answer.isdigit():
                answer = option[int(answer) - 1]
            processed_answers.append(answer)
        return processed_answers

    def create_cl_prompt(self, label_map, cl_set=None, round=0):
        if cl_set:
            labels = copy.deepcopy(cl_set)
            np.random.shuffle(labels)
        else:
            labels = np.random.choice(label_map, 4, replace=False)
        prompts = [
            f"[INST] <image>\nWhich label does not belong to this image? Answer the question with a single word from [{labels[0]}, {labels[1]}, {labels[2]}, {labels[3]}] [/INST]", 
            f"[INST] <image>\nWhich label does not belong to this image? (1) {labels[0]} (2) {labels[1]} (3) {labels[2]} (4) {labels[3]} Answer with the given letter directly [/INST]", 
            f"[INST] <image>\nWhich label does not belong to this image? (1) {labels[0]} (2) {labels[1]} (3) {labels[2]} (4) {labels[3]} Please provide your answer by stating the letter followed by the full option [/INST]", 
        ]
        return prompts[round], labels
    
    def create_ol_prompt(self, label_map, round=0, shuffle=False):
        labels = copy.deepcopy(label_map)
        if shuffle:
            np.random.shuffle(labels)
        labels_prompt_1 = ", ".join(labels)
        labels_prompt_2 = " ".join([f"({i + 1}) {labels[i]}" for i in range(len(labels))])
        prompts = [
            f"[INST] <image>\nWhich is the most related label to this image? Answer the question with a single word from [{labels_prompt_1}] [/INST]", 
            f"[INST] <image>\nWhich is the most related label to this image? {labels_prompt_2} Answer with the given letter directly [/INST]", 
            f"[INST] <image>\nWhich is the most related label to this image? {labels_prompt_2} Please provide your answer by stating the letter followed by the full option [/INST]", 
        ]
        return prompts[round], labels
