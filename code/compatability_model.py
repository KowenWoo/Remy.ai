'''
model to generate compatability embeddings
'''
import torch

class IngredientEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, ingredient_ids):
        return self.dropout(self.embeddings(ingredient_ids))

class CompatibilityModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.ingredient_embedding = IngredientEmbedding(vocab_size, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, ingredient1_ids, ingredient2_ids):
        emb1 = self.ingredient_embedding(ingredient1_ids)
        emb2 = self.ingredient_embedding(ingredient2_ids)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=-1)
        compatibility_score = self.classifier(combined)
        
        return compatibility_score, emb1, emb2