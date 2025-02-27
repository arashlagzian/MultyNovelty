import os
import random
import numpy as np
import nltk
import requests
import json
from nltk.tokenize import word_tokenize
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import argparse

# Ensure NLTK packages are downloaded
# nltk.download('punkt')




os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


prompts = [
    # 1.Philosophical Question:
    "What is the meaning of true happiness in life?",

    # 2.Hypothetical Scenario:
    "If humans could live on Mars, what challenges would they face and how could they overcome them?",

    # 3.Creative Thinking Prompt:
    "Can you describe an imaginary city where technology and nature exist in perfect harmony?",

    # 4.Practical Advice Question:
    "What are the most effective ways to learn a new language quickly?",

    # 5.Exploration of Abstract Concepts:
    "How would you explain the concept of time to someone who has never experienced it?",
    
    # 6.Scientific Exploration:
    "What are the possible effects of artificial intelligence on scientific research in the next decade?",

    # 7.Ethical Dilemma:
    "Is it ever justifiable to prioritize technological advancement over environmental protection?",

    # 8.Problem-Solving Question:
    "How can cities effectively reduce traffic congestion without compromising accessibility?",

    # 9.Imaginative Scenario:
    "If animals could communicate with humans, how would that change our world?",

    # 10.Personal Reflection Prompt:
    "What qualities make someone a great leader, and how can those qualities be developed?"
    ]


API_KEY = ""  # 
API_ENDPOINT = ""


def generate_outputs_api(prompt, num_samples, max_length):
    outputs = []
    
    for i in range(num_samples):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Your answers should be plain text without using special characters, bullet points, or lists."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_length,
            "temperature": 0.9,
            "top_p": 0.95
        }

        try:
            response = requests.post(API_ENDPOINT, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            output = response_data.get("choices")[0].get("message").get("content")
            outputs.append(output)

          
            print(f"\n[LOG] Prompt {i + 1}/{num_samples}: {prompt}")
            print(f"[LOG] Output {i + 1}/{num_samples}: {output}\n")
            
           
            if (i + 1) % 1 == 0:
                print(f"Generated {i + 1} outputs so far...")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response content: {response.content.decode('utf-8')}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text outputs using an API.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate per prompt.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of generated text.")
    args = parser.parse_args()


    num_samples = args.num_samples
    max_length = args.max_length

    print(f"Generating {num_samples} samples per prompt with max length {max_length}...")


    all_outputs = {}
    for prompt in prompts:
        outputs = generate_outputs_api(prompt, num_samples=num_samples, max_length=max_length)
        all_outputs[prompt] = outputs

    output_file = f"gptomini_{num_samples}{max_length}.json"
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=4)

    print(f"Outputs saved to {output_file}")


    for prompt, outputs in all_outputs.items():
        print(f"Prompt: {prompt}")
        for i, output in enumerate(outputs[:min(3, len(outputs))]):
            print(f"Output {i+1}: {output}")
        print("-" * 80)