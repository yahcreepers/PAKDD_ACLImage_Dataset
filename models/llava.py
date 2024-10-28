import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import numpy as np
import copy


class LLAVA:
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
    
    def predict(self, image, prompt):
        inputs = self.processor(prompt, image, padding=True, return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=30, pad_token_id=self.processor.tokenizer.eos_token_id)
        answers = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answers = [answer.split("[INST]")[-1].strip().replace(".", "").lower() for answer in answers]
        return answers

    def create_cl_prompt(self, label_map):
        random_label = np.random.choice(label_map, 4, replace=False)
        prompt = f"[INST] <image>\nWhich label does not belong to this image? Please answer with one word from [{random_label[0]}, {random_label[1]}, {random_label[2]}, {random_label[3]}] [INST]"
        # prompt = f"[INST] <image>\nWhat is the WRONG label of the picture that we provided in our question? Answer the question using a single label given in the provided list [{random_label[0]}, {random_label[1]}, {random_label[2]}, {random_label[3]}] [INST]"
        return prompt
    
    def create_ol_prompt(self, label_map):
        labels = copy.deepcopy(label_map)
        np.random.shuffle(labels)
        labels = ", ".join(labels)
        prompt = f"[INST] <image>\nWhich is the most related label to this image? Please answer with exactly one word from [{labels}] [INST]"
        return prompt