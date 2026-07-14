import torch

# Load one of your processed files
data = torch.load("C:/Users/Asus/Desktop/Jupiter's ballroom of Regret/vision_project/datasets/symbolic_cache_caltech256_split/train/processed_data.pt")

print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")  # You want to see torch.int64 or torch.long
print(f"Unique Values: {torch.unique(data)}") # Should be some subset of [0, 1, ..., 8]
print(f"Raw Grid Sample:\n{data[0, :5, :5]}") # Look at a 5x5 corner of the first item