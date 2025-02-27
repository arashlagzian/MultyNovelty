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
import torchvision 

# nltk.download('punkt')
#print(torch.__version__, torchvision.__version__)

set_seed(42)
random.seed(42)
np.random.seed(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


prompts = [
    # 1.Philosophical Question:
    "What is the meaning of true happiness in life?",

    # # 2.Hypothetical Scenario:
    "If humans could live on Mars, what challenges would they face and how could they overcome them?",

    # # 3.Creative Thinking Prompt:
    "Can you describe an imaginary city where technology and nature exist in perfect harmony?",

    # # 4.Practical Advice Question:
    "What are the most effective ways to learn a new language quickly?",

    # # 5.Exploration of Abstract Concepts:
    "How would you explain the concept of time to someone who has never experienced it?",
    
    # # 6.Scientific Exploration:
    "What are the possible effects of artificial intelligence on scientific research in the next decade?",

    # # 7.Ethical Dilemma:
    "Is it ever justifiable to prioritize technological advancement over environmental protection?",

    # # 8.Problem-Solving Question:
    "How can cities effectively reduce traffic congestion without compromising accessibility?",

    # # 9.Imaginative Scenario:
    "If animals could communicate with humans, how would that change our world?",

    # 10.Personal Reflection Prompt:
    "What qualities make someone a great leader, and how can those qualities be developed?"
    ]

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def build_prompt(system_message, user_message):
    """
    If you want to preserve a chat-style interface, you can build
    a custom prompt. You can omit the system message if you like
    and just pass the user prompt. 
    """
    prompt = f"System: {system_message}\nUser: {user_message}\nAssistant:"
    return prompt

def generate_outputs(prompts, num_samples, max_length, batch_size=100, temperature=0.9):
    outputs = {}
    for prompt in prompts:
        prompt_outputs = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_prompts = [prompt] * min(batch_size, num_samples - len(prompt_outputs))

            
            system_message = (
                            "You are a helpful assistant.Your answers should be plain text "
                            "without using special characters, bullet points, or lists. "
                            "Don't repeat the question in your answer and just provide the answer "
                            "at most 125 tokens."
                        )
            
            prompt_texts = [build_prompt(system_message, p) for p in batch_prompts]

            
            model_inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True
            ).to(device)

            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95
            )

            
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids):
                
                generated_ids_trimmed = output_ids[len(input_ids):]
                
                response = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
                prompt_outputs.append(response)

        outputs[prompt] = prompt_outputs
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text outputs from prompts.")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate per prompt.")
    parser.add_argument("--max_length", type=int, required=True,
                        help="Maximum length of generated text.")
    args = parser.parse_args()

    num_samples = args.num_samples
    max_length = args.max_length

    print(f"Generating {num_samples} samples with max length {max_length}...")

    
    outputs = generate_outputs(prompts, num_samples, max_length)

    
    output_file = f"deepseek_{num_samples}_{max_length}.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Outputs saved to {output_file}")

    # Display sample outputs
    #for prompt, generated_texts in outputs.items():
    #    print(f"Prompt: {prompt}")
    #    for i, text in enumerate(random.sample(generated_texts, min(3, len(generated_texts)))):
    #        print(f"Output {i+1}: {text}")
    #   print("-" * 80)