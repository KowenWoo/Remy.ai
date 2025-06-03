'''
clean and process dataset
'''

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re

def extract_ingr_rec(path):
    # Load the CSV file
    df = pd.read_csv(path)

    # Keep only the relevant columns
    df = df[["Title", "Ingredients", "Instructions"]]

    # Filter out rows where instructions are missing or empty
    df = df.dropna(subset=["Instructions"])
    df = df[df["Instructions"].str.strip() != ""]

    # Prepend title to the instructions
    df["Instructions"] = df["Title"].str.strip() + ": " + df["Instructions"].str.strip()

    # (Optional) Drop the title column if you no longer need it
    df = df.drop(columns=["Title"])
    return df

def setup_huggingface_model():
    """
    Set up a Hugging Face model for ingredient extraction
    Options:
    1. microsoft/DialoGPT-medium - Good for conversational tasks
    2. microsoft/DialoGPT-small - Faster, less memory
    3. google/flan-t5-base - Good for instruction following
    4. google/flan-t5-small - Smaller version
    """
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Option 1: T5 model (good for instruction following)
        model_name = "google/flan-t5-base"  # Change to "google/flan-t5-small" for faster processing
        
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create pipeline
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            max_length=512,
            do_sample=True,
            temperature=0.3
        )
        
        return generator
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a smaller model...")
        
        # Fallback to smaller model
        model_name = "google/flan-t5-small"
        generator = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=256,
            do_sample=True,
            temperature=0.3
        )
        
        return generator

def extract_ingredients_hf(ingredient_list, generator):
    """
    Extract core ingredients using Hugging Face model
    """
    try:
        # Convert list to string if needed
        if isinstance(ingredient_list, list):
            ingredients_text = "\n".join([f"- {item}" for item in ingredient_list])
        else:
            ingredients_text = str(ingredient_list)
        
        # Limit input length to avoid token limits
        if len(ingredients_text) > 1000:
            ingredients_text = ingredients_text[:1000] + "..."
        
        prompt = f"""Extract only the core ingredient names from this recipe list. Remove quantities, measurements, and adjectives.

Examples:
"2 cups butter, melted" -> "butter"
"1 large onion, diced" -> "onion"
"3 eggs" -> "eggs"

Ingredients:
{ingredients_text}

Core ingredients:"""

        # Generate response
        response = generator(prompt, max_length=200, num_return_sequences=1)
        result = response[0]['generated_text']
        
        # Clean up the result
        # Remove the original prompt from response if it's included
        if "Core ingredients:" in result:
            result = result.split("Core ingredients:")[-1]
        
        # Extract ingredient names
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Filter out common non-ingredients and clean up
        ingredients = []
        for line in lines:
            # Remove common prefixes/suffixes
            cleaned = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
            cleaned = re.sub(r'\d+\.\s*', '', cleaned)  # Remove numbers
            cleaned = cleaned.strip()
            
            # Skip empty lines or common non-ingredients
            if cleaned and len(cleaned) > 1 and not cleaned.lower() in ['the', 'and', 'or', 'with']:
                ingredients.append(cleaned)
        
        return ingredients if ingredients else [str(ingredient_list)]
        
    except Exception as e:
        print(f"Hugging Face model error: {e}")
        # Return original ingredient as fallback
        if isinstance(ingredient_list, list):
            return ingredient_list
        else:
            return [str(ingredient_list)]

# def extract_ingredients_regex_fallback(ingredient_text):
#     """
#     Simple regex-based extraction as fallback
#     """
#     if pd.isna(ingredient_text):
#         return []
    
#     # Convert to string
#     text = str(ingredient_text)
    
#     # Simple patterns to extract likely ingredients
#     # This is basic but works without API calls
#     ingredients = []
    
#     # Split by common separators
#     items = re.split(r'[,;]\s*', text)
    
#     for item in items:
#         # Remove quantities at the start
#         cleaned = re.sub(r'^\d+\.?\d*\s*(cups?|tbsp|tsp|lbs?|oz|grams?|kg)\s*', '', item, flags=re.IGNORECASE)
#         cleaned = re.sub(r'^\d+/\d+\s*', '', cleaned)  # Remove fractions
#         cleaned = re.sub(r'^\d+\s*', '', cleaned)  # Remove numbers
        
#         # Remove common adjectives and preparations
#         cleaned = re.sub(r'\b(large|small|medium|fresh|dried|chopped|diced|minced|sliced)\b', '', cleaned, flags=re.IGNORECASE)
        
#         cleaned = cleaned.strip(' ,-')
        
#         if len(cleaned) > 2:
#             ingredients.append(cleaned)
    
#     return ingredients

if __name__ == "__main__":
    path = "/Users/connectednorth/Documents/Remy.ai/data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

    df = extract_ingr_rec(path)
    print(f"Loaded {len(df)} recipes successfully")

    # Setup Hugging Face model
    print("Setting up Hugging Face model...")
    generator = setup_huggingface_model()
    use_hf_model = True
    print("Hugging Face model loaded successfully!")

    ingredients = df["Ingredients"].tolist()
    cleaned_ingredients = []
    
    for i, ingredient in enumerate(ingredients):
        if i % 50 == 0:  # Print progress every 50 items
            print(f"Processing recipe {i+1}/{len(ingredients)}")
        
        # Skip if ingredient is NaN or empty
        if pd.isna(ingredient) or str(ingredient).strip() == "":
            cleaned_ingredients.append([])
            continue
        
        cleaned = extract_ingredients_hf(ingredient, generator)


        cleaned_ingredients.append(cleaned)
            
        df["Cleaned_Ingredients"] = cleaned_ingredients
        df.to_csv("cleaned_recipes.csv", index=False)
        print("Cleaned ingredients saved to cleaned_recipes.csv")


