'''
clean and process dataset
'''

# import pandas as pd
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# import re

# def extract_ingr_rec(path):
#     # Load the CSV file
#     df = pd.read_csv(path)

#     # Keep only the relevant columns
#     df = df[["Title", "Ingredients", "Instructions"]]

#     # Filter out rows where instructions are missing or empty
#     df = df.dropna(subset=["Instructions"])
#     df = df[df["Instructions"].str.strip() != ""]

#     # Prepend title to the instructions
#     df["Instructions"] = df["Title"].str.strip() + ": " + df["Instructions"].str.strip()

#     # (Optional) Drop the title column if you no longer need it
#     df = df.drop(columns=["Title"])
#     return df

# def setup_huggingface_model():
#     """
#     Set up a Hugging Face model for ingredient extraction
#     Options:
#     1. microsoft/DialoGPT-medium - Good for conversational tasks
#     2. microsoft/DialoGPT-small - Faster, less memory
#     3. google/flan-t5-base - Good for instruction following
#     4. google/flan-t5-small - Smaller version
#     """
    
#     # Check if CUDA is available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     try:
#         # Option 1: T5 model (good for instruction following)
#         model_name = "google/flan-t5-base"  # Change to "google/flan-t5-small" for faster processing
        
#         print(f"Loading model: {model_name}")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)
        
#         # Create pipeline
#         generator = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             device=0 if device == "cuda" else -1,
#             max_length=512,
#             do_sample=True,
#             temperature=0.3
#         )
        
#         return generator
        
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Falling back to a smaller model...")
        
#         # Fallback to smaller model
#         model_name = "google/flan-t5-small"
#         generator = pipeline(
#             "text2text-generation",
#             model=model_name,
#             max_length=256,
#             do_sample=True,
#             temperature=0.3
#         )
        
#         return generator

# def extract_ingredients_hf(ingredient_list, generator):
#     """
#     Extract core ingredients using Hugging Face model
#     """
#     try:
#         # Convert list to string if needed
#         if isinstance(ingredient_list, list):
#             ingredients_text = "\n".join([f"- {item}" for item in ingredient_list])
#         else:
#             ingredients_text = str(ingredient_list)
        
#         # Limit input length to avoid token limits
#         if len(ingredients_text) > 1000:
#             ingredients_text = ingredients_text[:1000] + "..."
        
#         prompt = f"""Extract only the core ingredient names from this recipe list. Remove quantities, measurements, and adjectives.

# Examples:
# "2 cups butter, melted" -> "butter"
# "1 large onion, diced" -> "onion"
# "3 eggs" -> "eggs"

# Ingredients:
# {ingredients_text}

# Core ingredients:"""

#         # Generate response
#         response = generator(prompt, max_length=200, num_return_sequences=1)
#         result = response[0]['generated_text']
        
#         # Clean up the result
#         # Remove the original prompt from response if it's included
#         if "Core ingredients:" in result:
#             result = result.split("Core ingredients:")[-1]
        
#         # Extract ingredient names
#         lines = [line.strip() for line in result.split('\n') if line.strip()]
        
#         # Filter out common non-ingredients and clean up
#         ingredients = []
#         for line in lines:
#             # Remove common prefixes/suffixes
#             cleaned = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
#             cleaned = re.sub(r'\d+\.\s*', '', cleaned)  # Remove numbers
#             cleaned = cleaned.strip()
            
#             # Skip empty lines or common non-ingredients
#             if cleaned and len(cleaned) > 1 and not cleaned.lower() in ['the', 'and', 'or', 'with']:
#                 ingredients.append(cleaned)
        
#         return ingredients if ingredients else [str(ingredient_list)]
        
#     except Exception as e:
#         print(f"Hugging Face model error: {e}")
#         # Return original ingredient as fallback
#         if isinstance(ingredient_list, list):
#             return ingredient_list
#         else:
#             return [str(ingredient_list)]

# # def extract_ingredients_regex_fallback(ingredient_text):
# #     """
# #     Simple regex-based extraction as fallback
# #     """
# #     if pd.isna(ingredient_text):
# #         return []
    
# #     # Convert to string
# #     text = str(ingredient_text)
    
# #     # Simple patterns to extract likely ingredients
# #     # This is basic but works without API calls
# #     ingredients = []
    
# #     # Split by common separators
# #     items = re.split(r'[,;]\s*', text)
    
# #     for item in items:
# #         # Remove quantities at the start
# #         cleaned = re.sub(r'^\d+\.?\d*\s*(cups?|tbsp|tsp|lbs?|oz|grams?|kg)\s*', '', item, flags=re.IGNORECASE)
# #         cleaned = re.sub(r'^\d+/\d+\s*', '', cleaned)  # Remove fractions
# #         cleaned = re.sub(r'^\d+\s*', '', cleaned)  # Remove numbers
        
# #         # Remove common adjectives and preparations
# #         cleaned = re.sub(r'\b(large|small|medium|fresh|dried|chopped|diced|minced|sliced)\b', '', cleaned, flags=re.IGNORECASE)
        
# #         cleaned = cleaned.strip(' ,-')
        
# #         if len(cleaned) > 2:
# #             ingredients.append(cleaned)
    
# #     return ingredients

# if __name__ == "__main__":
#     path = "/Users/connectednorth/Documents/Remy.ai/data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

#     df = extract_ingr_rec(path)
#     print(f"Loaded {len(df)} recipes successfully")

#     # Setup Hugging Face model
#     print("Setting up Hugging Face model...")
#     generator = setup_huggingface_model()
#     use_hf_model = True
#     print("Hugging Face model loaded successfully!")

#     ingredients = df["Ingredients"].tolist()
#     cleaned_ingredients = []
    
#     for i, ingredient in enumerate(ingredients):
#         if i % 50 == 0:  # Print progress every 50 items
#             print(f"Processing recipe {i+1}/{len(ingredients)}")
        
#         # Skip if ingredient is NaN or empty
#         if pd.isna(ingredient) or str(ingredient).strip() == "":
#             cleaned_ingredients.append([])
#             continue
        
#         cleaned = extract_ingredients_hf(ingredient, generator)


#         cleaned_ingredients.append(cleaned)
            
#         df["Cleaned_Ingredients"] = cleaned_ingredients
#         df.to_csv("cleaned_recipes.csv", index=False)
#         print("Cleaned ingredients saved to cleaned_recipes.csv")


'''
GPU Cluster Parallelized Recipe Processing
Supports multi-node, multi-GPU processing with various parallelization strategies
'''

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import re
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from typing import List, Dict, Any
import time
import logging
from accelerate import Accelerator
from accelerate.utils import gather_object
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeDataset(Dataset):
    """Dataset class for recipe ingredients"""
    def __init__(self, ingredients_list):
        self.ingredients = ingredients_list
    
    def __len__(self):
        return len(self.ingredients)
    
    def __getitem__(self, idx):
        return {
            'idx': idx,
            'ingredients': self.ingredients[idx] if not pd.isna(self.ingredients[idx]) else ""
        }

class GPUClusterRecipeProcessor:
    def __init__(self, model_name="google/flan-t5-base", batch_size=8, max_length=512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        
    def setup_accelerate(self):
        """Setup Accelerate for multi-GPU/multi-node training"""
        self.accelerator = Accelerator()
        logger.info(f"Process {self.accelerator.process_index}: Using device {self.accelerator.device}")
        
    def setup_model(self):
        """Setup model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Handle different model types
        if "t5" in self.model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Move to device and wrap with accelerate
        if self.accelerator:
            self.model = self.accelerator.prepare(self.model)
        else:
            self.model = self.model.to(torch.cuda.current_device())
            
        self.model.eval()
        
    def create_prompt(self, ingredients_text):
        """Create extraction prompt"""
        if len(ingredients_text) > 1000:
            ingredients_text = ingredients_text[:1000] + "..."
            
        return f"""Extract only the core ingredient names from this recipe list. Remove quantities, measurements, and adjectives.

Examples:
"2 cups butter, melted" -> "butter"
"1 large onion, diced" -> "onion"
"3 eggs" -> "eggs"

Ingredients:
{ingredients_text}

Core ingredients:"""

    def process_batch(self, batch_ingredients):
        """Process a batch of ingredients"""
        results = []
        
        # Prepare prompts
        prompts = []
        valid_indices = []
        
        for i, ingredients in enumerate(batch_ingredients):
            if ingredients and str(ingredients).strip():
                prompts.append(self.create_prompt(str(ingredients)))
                valid_indices.append(i)
            else:
                results.append([])
        
        if not prompts:
            return results
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        if self.accelerator:
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=2,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode results
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process outputs
        batch_results = [[] for _ in range(len(batch_ingredients))]
        
        for i, (valid_idx, output) in enumerate(zip(valid_indices, decoded_outputs)):
            cleaned_ingredients = self.clean_output(output)
            batch_results[valid_idx] = cleaned_ingredients
        
        return batch_results
    
    def clean_output(self, output):
        """Clean model output to extract ingredients"""
        # Remove prompt if included in output
        if "Core ingredients:" in output:
            output = output.split("Core ingredients:")[-1]
        
        # Extract ingredient names
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        ingredients = []
        for line in lines:
            # Remove common prefixes/suffixes
            cleaned = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
            cleaned = re.sub(r'\d+\.\s*', '', cleaned)  # Remove numbers
            cleaned = cleaned.strip()
            
            # Skip empty lines or common non-ingredients
            if cleaned and len(cleaned) > 1 and cleaned.lower() not in ['the', 'and', 'or', 'with']:
                ingredients.append(cleaned)
        
        return ingredients if ingredients else []

def extract_ingr_rec(path):
    """Load and clean recipe dataset"""
    df = pd.read_csv(path)
    df = df[["Title", "Ingredients", "Instructions"]]
    df = df.dropna(subset=["Instructions"])
    df = df[df["Instructions"].str.strip() != ""]
    df["Instructions"] = df["Title"].str.strip() + ": " + df["Instructions"].str.strip()
    df = df.drop(columns=["Title"])
    return df

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

# Strategy 1: Accelerate-based Multi-GPU Processing
def process_with_accelerate(data_path, model_name="google/flan-t5-base", batch_size=8, output_path="cleaned_recipes_accelerate.csv"):
    """Process using Accelerate for automatic multi-GPU handling"""
    
    # Load data
    df = extract_ingr_rec(data_path)
    ingredients_list = df["Ingredients"].tolist()
    
    # Setup processor
    processor = GPUClusterRecipeProcessor(model_name, batch_size)
    processor.setup_accelerate()
    processor.setup_model()
    
    # Create dataset and dataloader
    dataset = RecipeDataset(ingredients_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: x  # Keep as list of dicts
    )
    
    # Prepare dataloader with accelerate
    dataloader = processor.accelerator.prepare(dataloader)
    
    # Process batches
    all_results = []
    
    with processor.accelerator.main_process_first():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                logger.info(f"Process {processor.accelerator.process_index}: Processing batch {batch_idx}")
            
            # Extract ingredients from batch
            batch_ingredients = [item['ingredients'] for item in batch]
            batch_results = processor.process_batch(batch_ingredients)
            
            # Store results with original indices
            for item, result in zip(batch, batch_results):
                all_results.append({
                    'idx': item['idx'],
                    'result': result
                })
    
    # Gather results from all processes
    all_results = gather_object(all_results)
    
    # Save results (only on main process)
    if processor.accelerator.is_main_process:
        # Sort by original index
        all_results.sort(key=lambda x: x['idx'])
        cleaned_ingredients = [item['result'] for item in all_results]
        
        df["Cleaned_Ingredients"] = cleaned_ingredients
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

# Strategy 2: Manual Distributed Processing
def process_distributed_worker(rank, world_size, data_path, model_name, batch_size, output_dir):
    """Worker function for manual distributed processing"""
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    try:
        # Load data
        df = extract_ingr_rec(data_path)
        ingredients_list = df["Ingredients"].tolist()
        
        # Split data across processes
        chunk_size = len(ingredients_list) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(ingredients_list)
        
        local_ingredients = ingredients_list[start_idx:end_idx]
        
        # Setup processor (without accelerate)
        processor = GPUClusterRecipeProcessor(model_name, batch_size)
        processor.setup_model()
        
        # Process local data
        results = []
        for i in range(0, len(local_ingredients), batch_size):
            batch = local_ingredients[i:i+batch_size]
            batch_results = processor.process_batch(batch)
            results.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Rank {rank}: Processed {i}/{len(local_ingredients)} items")
        
        # Save local results
        local_df = df.iloc[start_idx:end_idx].copy()
        local_df["Cleaned_Ingredients"] = results
        local_df.to_csv(f"{output_dir}/results_rank_{rank}.csv", index=False)
        
        logger.info(f"Rank {rank}: Completed processing {len(results)} items")
        
    finally:
        cleanup_distributed()

def launch_distributed_processing(data_path, model_name="google/flan-t5-base", batch_size=8, output_dir="distributed_results"):
    """Launch distributed processing across multiple GPUs"""
    
    world_size = torch.cuda.device_count()
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Launching distributed processing on {world_size} GPUs")
    
    mp.spawn(
        process_distributed_worker,
        args=(world_size, data_path, model_name, batch_size, output_dir),
        nprocs=world_size,
        join=True
    )
    
    # Combine results
    combine_distributed_results(output_dir, "cleaned_recipes_distributed.csv")

def combine_distributed_results(results_dir, output_path):
    """Combine results from distributed processing"""
    result_files = [f for f in os.listdir(results_dir) if f.startswith("results_rank_")]
    result_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    combined_df = pd.concat([
        pd.read_csv(os.path.join(results_dir, f)) 
        for f in result_files
    ], ignore_index=True)
    
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Combined results saved to {output_path}")

# Strategy 3: Hybrid CPU-GPU Processing
def process_hybrid_cpu_gpu(data_path, model_name="google/flan-t5-base", batch_size=8, num_cpu_workers=4, output_path="cleaned_recipes_hybrid.csv"):
    """Hybrid processing using CPU for preprocessing and GPU for inference"""
    
    def preprocess_worker(ingredients_batch):
        """CPU worker for preprocessing"""
        processed = []
        for ingredients in ingredients_batch:
            if pd.isna(ingredients) or str(ingredients).strip() == "":
                processed.append("")
            else:
                # Basic cleaning on CPU
                cleaned = str(ingredients)[:1000]  # Truncate
                processed.append(cleaned)
        return processed
    
    # Load data
    df = extract_ingr_rec(data_path)
    ingredients_list = df["Ingredients"].tolist()
    
    # Setup GPU processor
    processor = GPUClusterRecipeProcessor(model_name, batch_size)
    processor.setup_accelerate()
    processor.setup_model()
    
    # Process with CPU preprocessing
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_cpu_workers) as cpu_executor:
        # Process in chunks
        chunk_size = batch_size * 4  # Larger chunks for efficiency
        
        for i in range(0, len(ingredients_list), chunk_size):
            chunk = ingredients_list[i:i+chunk_size]
            
            # Preprocess on CPU
            future = cpu_executor.submit(preprocess_worker, chunk)
            preprocessed_chunk = future.result()
            
            # Process on GPU in smaller batches
            chunk_results = []
            for j in range(0, len(preprocessed_chunk), batch_size):
                batch = preprocessed_chunk[j:j+batch_size]
                batch_results = processor.process_batch(batch)
                chunk_results.extend(batch_results)
            
            all_results.extend(chunk_results)
            
            if i % (chunk_size * 5) == 0:
                logger.info(f"Processed {i}/{len(ingredients_list)} items")
    
    # Save results
    df["Cleaned_Ingredients"] = all_results
    df.to_csv(output_path, index=False)
    logger.info(f"Hybrid processing completed. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="GPU Cluster Recipe Processing")
    parser.add_argument("--data_path", required=True, help="Path to recipe CSV file")
    parser.add_argument("--model_name", default="google/flan-t5-base", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--strategy", choices=["accelerate", "distributed", "hybrid"], 
                       default="accelerate", help="Processing strategy")
    parser.add_argument("--output_path", default="cleaned_recipes.csv", help="Output path")
    parser.add_argument("--num_cpu_workers", type=int, default=4, help="CPU workers for hybrid mode")
    
    args = parser.parse_args()
    
    if args.strategy == "accelerate":
        process_with_accelerate(args.data_path, args.model_name, args.batch_size, args.output_path)
    elif args.strategy == "distributed":
        launch_distributed_processing(args.data_path, args.model_name, args.batch_size, "distributed_results")
    elif args.strategy == "hybrid":
        process_hybrid_cpu_gpu(args.data_path, args.model_name, args.batch_size, args.num_cpu_workers, args.output_path)

if __name__ == "__main__":
    main()