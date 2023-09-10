import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dims):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.ModuleList()
        for categories, dimension in embedding_dims:
            self.embeddings.append(nn.Embedding(categories, dimension))


    def forward(self, x):
        print("Inside Forward")
        x = [emb_layer(x[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        return x

