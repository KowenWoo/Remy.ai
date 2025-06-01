'''
clean and process dataset
'''

# import kagglehub
import pandas as pd
import regex as re
import spacy

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
    # df = df.drop(columns=["title"])
    return df

def clean_ingredient(ingredient_text):
    """
    Extract core ingredient name from a recipe ingredient description
    """
    # Convert to lowercase for processing
    text = ingredient_text.lower().strip()
    
    # Remove measurements and quantities at the beginning
    # Matches patterns like: "1", "2¾", "1 (3½–4-lb.)", "⅓", etc.
    text = re.sub(r'^[\d\s\(\)½¼¾⅓⅔⅛⅜⅝⅞\-–—\.]+(?:lb|oz|cup|cups|tsp|tbsp|tablespoon|teaspoon|pound|ounce|gallon|quart|pint|inch|inches|piece|pieces|clove|cloves)\.?\s*', '', text)
    text = re.sub(r'^[\d\s\(\)½¼¾⅓⅔⅛⅜⅝⅞\-–—\.]+', '', text)
    
    # Remove parenthetical descriptions like "(about 3 lb. total)" or "(such as Gala or Pink Lady)"
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove common descriptive adjectives and terms
    descriptive_words = [
        # Size/quantity descriptors
        'small', 'medium', 'large', 'big', 'tiny', 'whole', 'half', 'quarter', 'piece', 'pieces',
        'finely', 'coarsely', 'roughly', 'thinly', 'thickly', 'chopped', 'diced', 'sliced', 'minced',
        'crushed', 'ground', 'grated', 'shredded', 'torn', 'cut', 'cored', 'peeled', 'trimmed',
        
        # Quality/origin descriptors
        'fresh', 'organic', 'local', 'homemade', 'good-quality', 'high-quality', 'premium',
        'extra-virgin', 'virgin', 'cold-pressed', 'unfiltered', 'raw', 'natural', 'pure',
        'kosher', 'sea', 'coarse', 'fine', 'iodized',
        
        # Temperature/state descriptors
        'room temperature', 'melted', 'softened', 'frozen', 'dried', 'canned', 'bottled',
        'hot', 'cold', 'warm', 'cool', 'chilled',
        
        # Color descriptors
        'red', 'green', 'white', 'black', 'yellow', 'orange', 'purple', 'pink', 'brown',
        'golden', 'dark', 'light', 'pale', 'deep', 'bright',
        
        # Origin/style descriptors
        'italian', 'french', 'spanish', 'mexican', 'asian', 'chinese', 'japanese', 'thai',
        'indian', 'greek', 'moroccan', 'mediterranean', 'european', 'american',
        
        # Preparation descriptors
        'unsalted', 'salted', 'sweetened', 'unsweetened', 'seasoned', 'marinated',
        'smoked', 'grilled', 'roasted', 'baked', 'fried', 'steamed', 'boiled',
        
        # Additional descriptors
        'plus more', 'divided', 'or more', 'about', 'approximately', 'roughly',
        'sturdy', 'soft', 'hard', 'tender', 'crisp', 'smooth', 'chunky',
        'all-purpose', 'self-rising', 'bread', 'cake', 'pastry', 'tipo',
        'pinch of', 'dash of', 'splash of', 'drizzle of', 'handful of'
    ]
    
    # Create pattern to match descriptive words (word boundaries)
    pattern = r'\b(' + '|'.join(re.escape(word) for word in descriptive_words) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove extra punctuation and connecting words
    text = re.sub(r'\b(plus|and|or|with|for|to|into|from|such as|like)\b', '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces, commas, and punctuation
    text = re.sub(r'[,;]+', ' ', text)  # Replace commas/semicolons with spaces
    text = re.sub(r'\s+', ' ', text)     # Multiple spaces to single space
    text = text.strip(' ,.-')           # Remove leading/trailing punctuation
    
    # If we end up with something too short or empty, try to extract key nouns
    if len(text.strip()) < 3:
        # Try to find the main ingredient from the original text
        original_words = ingredient_text.lower().split()
        # Look for common ingredient words
        ingredient_keywords = [
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'turkey', 'lamb',
            'flour', 'sugar', 'salt', 'pepper', 'butter', 'oil', 'vinegar',
            'onion', 'garlic', 'carrot', 'potato', 'tomato', 'apple', 'lemon',
            'cheese', 'milk', 'cream', 'egg', 'bread', 'rice', 'pasta',
            'wine', 'broth', 'stock', 'water', 'miso', 'sage', 'rosemary',
            'squash', 'allspice', 'flakes'
        ]
        
        for word in original_words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in ingredient_keywords:
                return clean_word
    
    return text.strip() if text.strip() else ingredient_text

def process_ingredient_list(ingredient_list):
    """
    Process a list of ingredients and return cleaned ingredient names
    """
    if isinstance(ingredient_list, str):
        # If it's a string representation of a list, evaluate it
        try:
            ingredient_list = eval(ingredient_list)
        except:
            # If eval fails, treat as single ingredient
            return [clean_ingredient(ingredient_list)]
    
    cleaned_ingredients = []
    for ingredient in ingredient_list:
        cleaned = clean_ingredient(ingredient)
        if cleaned:  # Only add non-empty results
            cleaned_ingredients.append(cleaned)
    
    return cleaned_ingredients
# def extract_core_ingredient(ingredient_text, USE_SPACY, nlp=None):
#     # Load spaCy model if available
#     # First, remove common descriptive adjectives
#     adjectives_pattern = r'\b(fresh|organic|local|free-range|grass-fed|wild|farm|homemade|artisan|premium|quality|fine|pure|natural|raw|cooked|steamed|grilled|roasted|fried|baked|boiled|dried|frozen|canned|pickled|smoked|aged|young|old|new|hot|cold|warm|spicy|mild|sweet|sour|bitter|salty|tender|crisp|soft|hard|thick|thin|large|small|big|tiny|whole|half|quarter|sliced|diced|chopped|minced|ground|crushed|grated|shredded|red|green|white|black|yellow|orange|purple|pink|brown|blue|golden|dark|light|bright|pale|deep|rich|italian|french|spanish|mexican|asian|chinese|japanese|indian|thai|greek|moroccan|sicilian|tuscan|california|organic|gluten-free|sugar-free|fat-free|low-fat|non-fat|reduced|extra|super|ultra|double|triple)\b'
    
#     # Remove adjectives
#     cleaned = re.sub(adjectives_pattern, '', ingredient_text, flags=re.IGNORECASE)
    
#     # Clean up multiple spaces
#     cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
#     # If spaCy is available, use it for better noun extraction
#     if USE_SPACY and cleaned:
#         doc = nlp(cleaned)
#         nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
#         if nouns:
#             return nouns[-1]  # Return the last (usually main) noun
    
#     return cleaned if cleaned else ingredient_text

if __name__ == "__main__":

    # Download latest version
    # path = kagglehub.dataset_download("pes12017000148/food-ingredients-and-recipe-dataset-with-images")
    # print("Path to dataset files:", path)
    path = "/Users/connectednorth/Documents/Remy.ai/data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

    df = extract_ingr_rec(path)

    print(df["Ingredients"].iloc[1])
    test_ingredients = df["Ingredients"].iloc[1]

    cleaned = process_ingredient_list(test_ingredients)
    print(cleaned)

        # cleaned_ingred = []
        # cleaned_recipes = []
        # flag = 0
        # for ingredients in df["Ingredients"]:
        #     process_ingredient_list(ingredients)
        # df["Ingredients"] = cleaned_recipes

        # df.to_csv("dataset.csv", index=False)