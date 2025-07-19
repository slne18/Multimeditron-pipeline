from transformers import (
    FlaxVisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
from load_from_clip import load_model, encode_img
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import VisionTextDualEncoderProcessor
from PIL import Image
import json
import random
import os
import sys 

LINE_NUMBER = 300

random.seed(14)

#return the similarity between an image and a text according to the given model
def get_similarity(text, image_path, model, process):
    image = Image.open(image_path)
    inputs = process(text=[text], images=image, return_tensors="pt",truncation=True)

    outputs = model(**inputs)

    image_embeds = outputs.image_embeds 
    text_embeds = outputs.text_embeds    
    a_norm = torch.nn.functional.normalize(image_embeds, dim=1)
    b_norm = torch.nn.functional.normalize(text_embeds, dim=1)

    similarity = torch.matmul(a_norm, b_norm.T) 
    return similarity


def evaluate_model(model):
    clip_model = load_model(model)
    processor = VisionTextDualEncoderProcessor.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model.config.text_config._name_or_path)

    EVAL_DATASET = ""
    
    with open(EVAL_DATASET, "r", encoding="utf-8") as file:
        i = 0
       
        model_results = []

        lines = [l for u, l in enumerate(file) if u < LINE_NUMBER]
    
        good_guess = 0

        for line in lines:
            
            a = i
            b = i
            c = i
            while(a == i or b == i or c == i or a == b or a == c or b == c):
                a,b,c = random.sample(range(0,LINE_NUMBER),3)
            
            texts = []

            correct_line = json.loads(line)        
            a_line = json.loads(lines[a])
            b_line = json.loads(lines[b])
            c_line = json.loads(lines[c])
            
            texts.extend([correct_line["text"], a_line["text"], b_line["text"], c_line["text"]])
            model_similarities = []

            for t in texts:
                tokens = tokenizer.encode(t, truncation=True, max_length=500)
                text_value = tokenizer.decode(tokens, skip_special_tokens=True)
                image_value = correct_line["modalities"][0]['value']

                model_similarities.append(get_similarity(text_value,image_value, clip_model, processor).item())
          
            if(torch.argmax(torch.tensor(model_similarities)) == 0):
                good_guess = good_guess + 1
            i += 1
            if i >= LINE_NUMBER:
                break
    return good_guess/i


def main():
    #evaluated image encoders
    clips = [("standard_clip", "openai/clip-vit-base-patch32")]

    for model in clips:
        print(model[0] + " accuracy : " + str(evaluate_model(model[1])) + "\n")       

if __name__ == "__main__":
    main()