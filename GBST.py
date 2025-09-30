# Handles all the logic for GBST, vectorisation, sub-blocking, mean pooling etc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class GBST(nn.Module):
    def __init__(self, embedding_dim, block_size, num_candidates):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.num_candidates = num_candidates

        # A small linear layer to produce weights for each candidate
        self.weight_layer = nn.Linear(embedding_dim, num_candidates)

    def vectorisation(self, text, embedding_dim):
        Length = len(text) + 2  # length of text with padding

        vectors = torch.arange(Length * embedding_dim, dtype=torch.float32).reshape(Length, embedding_dim)

        return vectors

    # Helper Method
    def _make_blocks(self, vectors, block_size):
        n_blocks = vectors.size(0) // block_size  # truncates incase indivisibility
        blocks = vectors[:n_blocks * block_size].reshape(n_blocks, block_size, -1)
        return blocks # (num_blocks, block_size, embedding_dim)

    def meanPooling(self, blocks):
        # average across the block_size dimension
        return blocks.mean(dim=1)  # (num_blocks, embedding_dim)

    def softSelection(self, vectors):
        candidates = []
        min_blocks = None

        for span in range(2, self.num_candidates + 2):  # spans=2,3,4
            blocks = self._make_blocks(vectors, span)  # (num_blocks, span, D)
            print(f"Span={span}, blocks.shape={blocks.shape}")
            pooled = blocks.mean(dim=1)  # (num_blocks, D)

            if min_blocks is None:
                min_blocks = pooled.size(0)
            else:
                min_blocks = min(min_blocks, pooled.size(0))

            candidates.append(pooled)

        # Truncate all candidates to the same min length
        candidates = [c[:min_blocks] for c in candidates]

        candidates = torch.stack(candidates, dim=1)  # (min_blocks, num_candidates, D)

        logits = self.weight_layer(candidates[:, 0, :])  # (min_blocks, num_candidates)
        weights = F.softmax(logits, dim=-1)

        print("Weights:", weights.detach().cpu().numpy())

        weighted = torch.einsum("bn,bnd->bd", weights, candidates)  # (min_blocks, D)
        return weighted

    def downsample(self, x, stride):
        # x: (num_blocks, D)
        n = x.size(0) // stride
        x = x[:n * stride].reshape(n, stride, -1)  # (n, stride, D)
        return x.mean(dim=1)  # (n, D)
