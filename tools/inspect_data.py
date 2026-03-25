import os
import glob
import numpy as np
import h5py
import nibabel as nib

def inspect_dir(path, name):
    print(f"\n--- Inspecting {name}: {path} ---")
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return

    files = os.listdir(path)
    print(f"Found {len(files)} files/dirs.")
    
    # Sample a few files
    sample_files = files[:5]
    for f in sample_files:
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            print(f"Dir: {f}")
        else:
            ext = os.path.splitext(f)[1].lower()
            print(f"File: {f} (Ext: {ext})")
            
            try:
                if ext == '.npy':
                    data = np.load(full_path)
                    print(f"  Shape: {data.shape}, Dtype: {data.dtype}, Range: [{data.min()}, {data.max()}]")
                elif ext in ['.h5', '.hdf5']:
                    with h5py.File(full_path, 'r') as hf:
                        print(f"  Keys: {list(hf.keys())}")
                        # Peek at first key
                        k = list(hf.keys())[0]
                        print(f"  Shape of '{k}': {hf[k].shape}")
                elif ext in ['.nii', '.gz']:
                    img = nib.load(full_path)
                    print(f"  Shape: {img.shape}")
            except Exception as e:
                print(f"  Error reading: {e}")

paths = [
    r"Z:\dataset\M4Raw_multicoil_test",
    r"Z:\dataset\M4RawV1.5_gre_data",
    r"Z:\dataset\M4RawV1.5_motion",
    r"Z:\dataset\M4RawV1.5_multicoil_train",
    r"Z:\dataset\M4RawV1.5_multicoil_val",
    r"Z:\dataset\Sparsity_SDOCT_DATASET_2012"
]

for p in paths:
    inspect_dir(p, os.path.basename(p))
