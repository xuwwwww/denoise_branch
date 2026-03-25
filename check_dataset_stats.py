import numpy as np
import os

data_root = 'dataset'
files = {
    'CT Train (Patch)': os.path.join(data_root, 'cmayo/train_64.npy'),
    'CT Test (Full)': os.path.join(data_root, 'cmayo/test_512.npy'),
    'MRI (Appendix)': os.path.join(data_root, 'cmayo_appendix/mri_appendix.npy'),
    'BUSI (Appendix)': os.path.join(data_root, 'cmayo_appendix/busi_appendix.npy')
}

print("Dataset Statistics:")
print("-" * 60)
print(f"{'Dataset':<20} | {'Shape':<20} | {'Count':<10} | {'Size':<10}")
print("-" * 60)

for name, path in files.items():
    try:
        if os.path.exists(path):
            # Load only metadata if possible to save memory, but npy usually loads all.
            # Using mmap_mode='r' to avoid loading into RAM
            data = np.load(path, mmap_mode='r')
            shape = data.shape
            # Shape is usually [2, N, H, W] or [N, 2, H, W] depending on save format.
            # dataset.py says: data[0], data[1] -> so it's [2, N, H, W]
            
            if shape[0] == 2:
                count = shape[1]
                size = f"{shape[2]}x{shape[3]}"
            else:
                # Maybe [N, 2, H, W]
                count = shape[0]
                size = f"{shape[2]}x{shape[3]}" # Assuming 2nd dim is channel/pair
                
            print(f"{name:<20} | {str(shape):<20} | {count:<10} | {size:<10}")
        else:
            print(f"{name:<20} | {'Not Found':<20} | {'-':<10} | {'-':<10}")
    except Exception as e:
        print(f"{name:<20} | {str(e):<20} | {'Error':<10} | {'-':<10}")

print("-" * 60)
