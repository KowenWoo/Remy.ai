'''
generate relationality between ingredients commonly found in recipes together
'''

import torch
import random
import numpy as np
import matplotlib as plt

def generate_positive_pairs(recipes):
    positive_pairs = []
    
    for recipe in recipes:
        ingredients = recipe['ingredients']
        
        # All ingredient pairs in same recipe
        for i in range(len(ingredients)):
            for j in range(i+1, len(ingredients)):
                pair_data = {
                    'ingredient_1': ingredients[i],
                    'ingredient_2': ingredients[j],
                    'label': 1,  # Compatible
                    'recipe_count': 1
                }
                positive_pairs.append(pair_data)
    
    # Aggregate by ingredient pair
    pair_counts = defaultdict(list)
    for pair in positive_pairs:
        key = tuple(sorted([pair['ingredient_1'], pair['ingredient_2']]))
        pair_counts[key].append(pair)
    
    # Create final positive dataset
    final_positive = []
    for pair_key, pair_list in pair_counts.items():
        final_positive.append({
            'ingredient_1': pair_key[0],
            'ingredient_2': pair_key[1],
            'label': 1,
            'co_occurrence_count': len(pair_list),
            'avg_rating': np.mean([p['avg_rating'] for p in pair_list]),
            'cuisines': list(set([p['cuisine'] for p in pair_list]))
        })
    
    return final_positive


def generate_negative_pairs(recipes, positive_pairs, ratio=0.3):
    # Get all unique ingredients
    all_ingredients = set()
    for recipe in recipes:
        all_ingredients.update(recipe['ingredients'])
    
    # Create set of positive pairs for quick lookup
    positive_set = set()
    for pair in positive_pairs:
        key = tuple(sorted([pair['ingredient_1'], pair['ingredient_2']]))
        positive_set.add(key)
    
    negative_pairs = []
    target_count = int(len(positive_pairs) * ratio)
    
    while len(negative_pairs) < target_count:
        # Sample two random ingredients
        ing1, ing2 = random.sample(list(all_ingredients), 2)
        pair_key = tuple(sorted([ing1, ing2]))
        
        # If they never appear together, it's a negative pair
        if pair_key not in positive_set:
            negative_pairs.append({
                'ingredient_1': ing1,
                'ingredient_2': ing2,
                'label': 0,  # Incompatible
                'co_occurrence_count': 0
            })
    
    return negative_pairs

def add_domain_negatives(negative_pairs):
    for combo in bad_combinations:
        negative_pairs.append({
            'ingredient_1': combo[0],
            'ingredient_2': combo[1], 
            'label': 0,
            'co_occurrence_count': 0,
            'confidence': 1.0  # High confidence these are bad
        })
    return negative_pairs

def train_compatibility_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            ingredient1_ids = batch['ingredient1_ids']
            ingredient2_ids = batch['ingredient2_ids']
            labels = batch['labels'].float()
            
            compatibility_scores, emb1, emb2 = model(ingredient1_ids, ingredient2_ids)
            
            # Main compatibility loss
            compat_loss = criterion(compatibility_scores.squeeze(), labels)
            
            # Optional: Add embedding regularization
            reg_loss = embedding_regularization(emb1, emb2)
            
            total_loss = compat_loss + 0.01 * reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Validation
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Val Acc={val_acc:.4f}")

def embedding_regularization(emb1, emb2):
    # Encourage similar ingredients to have similar embeddings
    # This could be based on ingredient categories, cuisine types, etc.
    return torch.mean(torch.norm(emb1, dim=1)) + torch.mean(torch.norm(emb2, dim=1))

def evaluate_embeddings(model, test_pairs):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_pairs:
            scores, _, _ = model(batch['ingredient1_ids'], batch['ingredient2_ids'])
            predictions = (scores > 0.5).float().squeeze()
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
    
    accuracy = correct / total
    return accuracy

def test_ingredient_similarity(model, ingredient_vocab):
    # Test cases
    test_cases = [
        ("chicken", ["beef", "pork", "fish", "apple"]),  # Should rank proteins higher
        ("basil", ["oregano", "thyme", "carrot", "chocolate"]),  # Should rank herbs higher
        ("tomato", ["onion", "garlic", "ice_cream", "vanilla"])  # Should rank vegetables higher
    ]
    
    for base_ingredient, candidates in test_cases:
        base_emb = get_ingredient_embedding(model, base_ingredient)
        similarities = []
        
        for candidate in candidates:
            cand_emb = get_ingredient_embedding(model, candidate)
            sim = F.cosine_similarity(base_emb, cand_emb, dim=0)
            similarities.append((candidate, sim.item()))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"{base_ingredient}: {similarities}")

def visualize_embeddings(model, ingredient_vocab, sample_size=500):
    # Get embeddings for visualization
    embeddings = []
    labels = []
    
    for ingredient in random.sample(list(ingredient_vocab.keys()), sample_size):
        emb = get_ingredient_embedding(model, ingredient)
        embeddings.append(emb.cpu().numpy())
        labels.append(ingredient)
    
    # Reduce dimensions for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Annotate some points
    for i, label in enumerate(labels[:50]):  # Show first 50 labels
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("Ingredient Embeddings Visualization")
    plt.show()


def save_ingredient_embeddings(model, ingredient_vocab, save_path):
    # Extract just the embedding weights
    embedding_weights = model.ingredient_embedding.embeddings.weight.data
    
    # Save with vocabulary mapping
    torch.save({
        'embedding_weights': embedding_weights,
        'vocab_to_id': ingredient_vocab,
        'id_to_vocab': {v: k for k, v in ingredient_vocab.items()},
        'embedding_dim': embedding_weights.shape[1]
    }, save_path)

    # Save the model
    save_ingredient_embeddings(model, ingredient_vocab, 'ingredient_embeddings.pth')

if __name__ == "__main__":
    #get data 

    #generate relationships

    #train model
