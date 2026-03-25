import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from models import model_dict, TrainTask

# Monkey patch set_loader to avoid loading training data which might be missing
def no_op_set_loader(self):
    print("Skipping default data loader initialization...")
    pass

TrainTask.set_loader = no_op_set_loader

def visualize(opt, image_index=0):
    # 5. Load Data
    print("Loading data from test_512.npy...")
    try:
        data = np.load("dataset/test_512.npy")
        print(f"Dataset shape: {data.shape}")
        # data[0] is low dose, data[1] is full dose
        full_dose_np = data[1][image_index]
        print(f"Image shape: {full_dose_np.shape}")
        
    except Exception as e:
        print(f"Error loading npy: {e}")
        return

    # --- Plot Golden (Ground Truth) ---
    print("Generating Golden Image...")
    plt.figure(figsize=(10, 10))
    plt.title("Golden (Ground Truth)")
    plt.imshow(full_dose_np, cmap='gray')
    plt.axis('off')
    
    if not os.path.exists("visualization"):
        os.makedirs("visualization")
        
    save_path_golden = f"visualization/result_Golden.png"
    plt.savefig(save_path_golden, bbox_inches='tight', pad_inches=0)
    print(f"Saved Golden visualization to {save_path_golden}")
    plt.close()

if __name__ == '__main__':
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    
    # Fix default model name if not provided
    if default_opt.model_name == 'supcon':
        default_opt.model_name = 'DUGAN'
        
    # We don't need to load the model for Golden image, just options
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    
    visualize(opt)
