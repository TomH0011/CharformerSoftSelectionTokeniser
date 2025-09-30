from config import embedding_dim, text, block_size, num_candidates, stride
from GBST import GBST

class Main:
    def __init__(self):
        self.text = text
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.num_candidates = num_candidates
        self.stride = stride

        self.gbst = GBST(
            embedding_dim=self.embedding_dim,
            block_size=self.block_size,
            num_candidates=self.num_candidates
        )

    def run(self):
        # Vectorise characters
        vectors = self.gbst.vectorisation(self.text, self.embedding_dim)

        # Soft selection over candidate spans
        selected = self.gbst.softSelection(vectors)

        # Downsample to reduce sequence length
        downsampled = self.gbst.downsample(selected, self.stride)

        return downsampled


if __name__ == "__main__":
    model = Main()
    out = model.run()
    print("Output shape:", out.shape)
