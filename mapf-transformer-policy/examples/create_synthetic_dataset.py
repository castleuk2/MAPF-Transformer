from pathlib import Path

from mapf_transformer.synthetic import create_synthetic_dataset

if __name__ == "__main__":
    manifest = create_synthetic_dataset(Path("data/synthetic"), episodes=8, seed=7)
    print(manifest)
