import os
import random
import numpy as np
import nltk
import json
from nltk.tokenize import word_tokenize
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import argparse

# Ensure NLTK packages are downloaded
# nltk.download('punkt')

# Set seed for reproducibility
set_seed(42)
random.seed(42)
np.random.seed(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "2"  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

prompts = [
    "What is the meaning of true happiness in life?",
    "If humans could live on Mars, what challenges would they face and how could they overcome them?",
    "Can you describe an imaginary city where technology and nature exist in perfect harmony?",
    "What are the most effective ways to learn a new language quickly?",
    "How would you explain the concept of time to someone who has never experienced it?",
    "What are the possible effects of artificial intelligence on scientific research in the next decade?",
    "Is it ever justifiable to prioritize technological advancement over environmental protection?",
    "How can cities effectively reduce traffic congestion without compromising accessibility?",
    "If animals could communicate with humans, how would that change our world?",
    "What qualities make someone a great leader, and how can those qualities be developed?"
]


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


def generate_outputs(prompts, num_samples, max_length, batch_size=100, temperature=0.9):
    outputs = {}
    for prompt in prompts:
        prompt_outputs = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
           
            batch_prompts = [prompt] * min(batch_size, num_samples - len(prompt_outputs))

          
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
            texts = [
                tokenizer.apply_chat_template(
                    messages + [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for p in batch_prompts
            ]

           
            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True
            ).to(model.device)

           
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95
            )

           
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                generated_ids_trimmed = output_ids[len(input_ids):]  
                response = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
                prompt_outputs.append(response)

        outputs[prompt] = prompt_outputs
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text outputs from prompts.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate per prompt.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of generated text.")
    args = parser.parse_args()

    num_samples = args.num_samples
    max_length = args.max_length

    print(f"Generating {num_samples} samples with max length {max_length}...")


    outputs = generate_outputs(prompts, num_samples, max_length)

  
    output_file = f"qwen_{num_samples}{max_length}.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Outputs saved to {output_file}")

 
    for prompt, generated_texts in outputs.items():
        print(f"Prompt: {prompt}")
        for i, text in enumerate(random.sample(generated_texts, min(3, len(generated_texts)))):
            print(f"Output {i+1}: {text}")
        print("-" * 80)
