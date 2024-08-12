import pickle as pkl

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    print(f"Dataset loaded from {file_path}")
    return dataset
