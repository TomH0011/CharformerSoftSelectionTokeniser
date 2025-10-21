import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import importlib
import sys
import os
import config
from GBST import GBST

# Force reload config to get latest values 
__pycache__ = os.path.join(os.path.dirname(__file__), '__pycache__')
if os.path.exists(__pycache__):
    for file in os.listdir(__pycache__):
        if file.startswith('config'):
            try:
                os.remove(os.path.join(__pycache__, file))
            except:
                pass

# remove from sys.modules to force fresh import
for module in list(sys.modules.keys()):
    if module == 'config' or module.startswith('config.'):
        del sys.modules[module]


class GBSTVisualiser:
    def __init__(self):
        self.text = config.text
        self.embedding_dim = config.embedding_dim
        self.block_size = config.block_size
        self.num_candidates = config.num_candidates
        self.stride = config.stride

        self.gbst = GBST(
            embedding_dim=self.embedding_dim,
            block_size=self.block_size,
            num_candidates=self.num_candidates
        )

        # Storage for visualisation data
        self.block_weights = None
        self.candidate_blocks = None
        self.min_blocks = None
        self.original_text_positions = None

    def visualise_soft_selection(self, vectors):
        candidates = []
        min_blocks = None

        for span in range(2, self.num_candidates + 2):  # spans=2,3,4
            blocks = self.gbst._make_blocks(vectors, span)
            print(f"Span={span}, blocks.shape={blocks.shape}")
            pooled = blocks.mean(dim=1)

            if min_blocks is None:
                min_blocks = pooled.size(0)
            else:
                min_blocks = min(min_blocks, pooled.size(0))

            candidates.append(pooled)

        # Truncate all candidates to the same min length
        candidates = [c[:min_blocks] for c in candidates]
        candidates_stacked = torch.stack(candidates, dim=1)  # (min_blocks, num_candidates, D)

        # Compute weights
        logits = self.gbst.weight_layer(candidates_stacked[:, 0, :])
        weights = F.softmax(logits, dim=-1)

        # Store for visualisation
        self.block_weights = weights.detach().cpu().numpy()
        self.min_blocks = min_blocks
        self.candidate_blocks = candidates_stacked.detach().cpu().numpy()

        print("Weights shape:", self.block_weights.shape)
        print("Weights:\n", self.block_weights)

        # Compute weighted combination
        weighted = torch.einsum("bn,bnd->bd", weights, candidates_stacked)
        return weighted

    def create_heatmaps(self):
        vectors = self.gbst.vectorisation(self.text, self.embedding_dim)
        selected = self.visualise_soft_selection(vectors)
        downsampled = self.gbst.downsample(selected, self.stride)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))

        # Main heatmap, block weights for each candidate span
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(self.block_weights,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    xticklabels=[f'Span {i + 2}' for i in range(self.num_candidates)],
                    yticklabels=[f'Block {i}' for i in range(self.min_blocks)],
                    cbar_kws={'label': 'Attention Weight'},
                    ax=ax1)
        ax1.set_title('Block-wise Candidate Span Weights\n(Which span size each block prefers)',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Candidate Span size', fontsize=10)
        ax1.set_ylabel('Block Index', fontsize=10)

        # Block importance (sum of weights across candidates)
        ax2 = plt.subplot(2, 2, 2)
        block_importance = self.block_weights.sum(axis=1, keepdims=True)
        sns.heatmap(block_importance,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    yticklabels=[f'Block {i}' for i in range(self.min_blocks)],
                    xticklabels=['Importance'],
                    cbar_kws={'label': 'Total Importance'},
                    ax=ax2)
        ax2.set_title('Block Total Importance\n(Sum across all candidates)',
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel('Block Index', fontsize=10)

        # Dominant span per block
        ax3 = plt.subplot(2, 2, 3)
        dominant_span = np.argmax(self.block_weights, axis=1)
        max_weights = np.max(self.block_weights, axis=1)

        # Create a heatmap showing which span is dominant
        dominant_data = dominant_span.reshape(-1, 1)
        sns.heatmap(dominant_data,
                    annot=[[f'Span {d + 2}\n({max_weights[i]:.3f})']
                           for i, d in enumerate(dominant_span)],
                    fmt='',
                    cmap='Set2',
                    yticklabels=[f'Block {i}' for i in range(self.min_blocks)],
                    xticklabels=['Dominant Span'],
                    cbar_kws={'label': 'Span size (2-4)'},
                    ax=ax3)
        ax3.set_title('Dominant Span per Block\n(Most preferred span size)',
                      fontsize=12, fontweight='bold')
        ax3.set_ylabel('Block Index', fontsize=10)

        # Span preference distribution
        ax4 = plt.subplot(2, 2, 4)
        span_preference = self.block_weights.mean(axis=0)
        bars = ax4.bar(range(self.num_candidates), span_preference,
                       color=['#ff9999', '#66b3ff', '#99ff99'])
        ax4.set_xticks(range(self.num_candidates))
        ax4.set_xticklabels([f'Span {i + 2}' for i in range(self.num_candidates)])
        ax4.set_ylabel('Average Weight', fontsize=10)
        ax4.set_title('Overall Span Preference\n(Average across all blocks)',
                      fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, span_preference)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.3f}',
                     ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'GBST Charformer Tokeniser - Block Importance Analysis\nText: "{self.text}"',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('gbst_block_importance_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n[SUCCESS] Heatmap saved as 'gbst_block_importance_heatmap.png'")
        plt.close()

        return downsampled

    def create_detailed_text_mapping(self):
        vectors = self.gbst.vectorisation(self.text, self.embedding_dim)
        selected = self.visualise_soft_selection(vectors)

        fig, axes = plt.subplots(self.num_candidates, 1, figsize=(14, 8))

        for idx, span in enumerate(range(2, self.num_candidates + 2)):
            ax = axes[idx] if self.num_candidates > 1 else axes

            # Create blocks for this span
            blocks = self.gbst._make_blocks(vectors, span)
            num_blocks = blocks.shape[0]

            # Get weights for this span across all blocks (only up to min_blocks)
            weights_for_span = self.block_weights[:, idx]

            # Create visualisation only for blocks that have weights
            block_positions = np.arange(self.min_blocks)
            colors = plt.cm.YlOrRd(weights_for_span / (weights_for_span.max() + 1e-8))

            bars = ax.barh(block_positions, weights_for_span, color=colors, edgecolor='black')

            # Add text labels showing which characters are in each block
            text_with_padding = f"<{self.text}>"  # Simulating padding
            for i in range(self.min_blocks):
                start_char = i * span
                end_char = start_char + span
                if end_char <= len(text_with_padding):
                    char_slice = text_with_padding[start_char:end_char]
                    ax.text(weights_for_span[i] * 0.02, i, f' "{char_slice}"',
                            va='center', ha='left', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_ylabel('Block Index', fontsize=10)
            ax.set_xlabel('Attention Weight', fontsize=10)
            ax.set_title(f'Span size {span} - Block Importance with Text Mapping', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()

        plt.suptitle(f'Text-to-Block Mapping Analysis\nText: "{self.text}"',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gbst_text_block_mapping.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Text mapping saved as 'gbst_text_block_mapping.png'")
        plt.close()


if __name__ == "__main__":
    visualiser = GBSTVisualiser()

    print("=" * 60)
    print("GBST CHARFORMER BLOCK IMPORTANCE VISUALISATION")
    print("=" * 60)

    # Create main heatmap
    output = visualiser.create_heatmaps()
    print(f"\nOutput shape: {output.shape}")

    print("\n" + "=" * 60)

    # Create detailed text-to-block mapping
    visualiser.create_detailed_text_mapping()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"- Total blocks analysed: {visualiser.min_blocks}")
    print(f"- Candidate spans tested: {visualiser.num_candidates} (sizes: 2, 3, 4)")
    print(f"- Input text: '{config.text}'")
    print(f"- Most important block: Block {np.argmax(visualiser.block_weights.sum(axis=1))}")
    print(f"- Most preferred span overall: Span {np.argmax(visualiser.block_weights.mean(axis=0)) + 2}")
    print("=" * 60)
