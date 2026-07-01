import os
import json
import random
import argparse

def generate_split(data_dir, total_files, train_p, val_p, test_p, output_name, seed=42):
    all_files = os.listdir(data_dir)
    raw_ids = sorted(list(set([f.split('_')[0] for f in all_files if f.endswith('.npy')])))
    patients = [f"paciente{pid.zfill(4)}" for pid in raw_ids]
    
    random.seed(seed)
    random.shuffle(patients)
    selected_patients = patients[:total_files]
    
    n = len(selected_patients)
    n_train = int(n * train_p)
    n_val = int(n * val_p)
    
    train_set = selected_patients[:n_train]
    val_set = selected_patients[n_train:n_train + n_val]
    test_set = selected_patients[n_train + n_val:]
    
    split_data = {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
    
    with open(f"{output_name}", 'w') as f:
        json.dump(split_data, f, indent=4)
    
    print(f"JSON gerado: {output_name} com {len(train_set)} treino, {len(val_set)} val, {len(test_set)} teste.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../dataset/new_images")
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--train_p", type=float, default=0.7)
    parser.add_argument("--val_p", type=float, default=0.15)
    parser.add_argument("--test_p", type=float, default=0.15)
    parser.add_argument("--output", default="my_dataset_split.json")
    
    args = parser.parse_args()
    generate_split(args.data_dir, args.total, args.train_p, args.val_p, args.test_p, args.output)