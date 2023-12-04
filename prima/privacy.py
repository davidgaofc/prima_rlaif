import torch
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random

class PriMa():
    def __init__(self, huggingface_model="google/bert_uncased_L-2_H-128_A-2", prop=0.3, passes=1):
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.model = AutoModelForMaskedLM.from_pretrained(huggingface_model)
        self.prop = prop
        self.passes = passes

    #augments RLAIF prompt
    def augment(self, text):
        split = self.separate_text(text)
        assert len(split) == 3
        for i in range(self.passes):
            for i in range(len(split)):
                my_map = {0,1,2}
                my_map.remove(i)
                split[i] = self.mask_sequence(split[i])
                joined = " ".join(split)
                new = self.fill_masks(joined)
                my_map = list(my_map)
                rm1 = split[my_map[0]].lower()
                rm2 = split[my_map[1]].lower()
                new = new.replace(rm1, "")
                new = new.replace(rm2, "")
                split[i] = new
        return "human: " + split[0] + "\nassistant:\n" + " 1. " + split[1] + " \n2. " + split[2]

    def fill_masks(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt")
        mask_token_indices = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(**inputs).logits
        for mask_index in mask_token_indices:
            predicted_token_id = token_logits[0, mask_index].argmax(axis=-1)
            inputs.input_ids[0, mask_index] = predicted_token_id
        return self.tokenizer.decode(inputs.input_ids[0][1:-1])

    #masks sequence with [MASK] tokens for certain proportion of words and generates new text
    def mask_sequence(self, sequence):
        words = sequence.split()
        for i in range(len(words)):
            if random.random() < self.prop:
                words[i] = "[MASK]"
        return " ".join(words)

    #separates RLAIF text into prompt, answer1, and answer2
    def separate_text(self, text):
        parts = re.split(r'(human:|\nassistant:\n1\.|\n2\.)', text)
        desired_parts = []
        for part in parts:
            stripped_part = part.strip()
            if stripped_part and stripped_part not in ['human:', '\nassistant:\n1.', '\n2.', "assistant:\n1.", "2."]:
                desired_parts.append(stripped_part)
        if(len(desired_parts) != 3):
            print("Warning: text not in correct format")
            desired_parts.append(" ")
        return desired_parts

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    model = AutoModelForMaskedLM.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    prop = 0.3
    passes = 1
