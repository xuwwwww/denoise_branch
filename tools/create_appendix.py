import os
import numpy as np
import h5py
import glob
from tqdm import tqdm
import torch
from PIL import Image

def normalize(data):
    """Normalize data to [0, 1] range."""
    d_min = data.min()
    d_max = data.max()
    if d_max - d_min < 1e-6:
        return data
    return (data - d_min) / (d_max - d_min)

def read_h5_volume(filepath):
    """Read volume from h5 file using heuristic for keys."""
    with h5py.File(filepath, 'r') as hf:
        keys = list(hf.keys())
        # M4Raw keys often include 'kspace', 'reconstruction_rss', 'target', 'input'
        # We prefer reconstruction or target
        preferred_keys = ['reconstruction_rss', 'reconstruction_esc', 'target', 'image', 'vol']
        
        selected_key = None
        for k in preferred_keys:
            if k in keys:
                selected_key = k
                break
        
        if selected_key is None:
            # Fallback to first key that looks like data (ndim >= 2)
            for k in keys:
                if hasattr(hf[k], 'shape') and len(hf[k].shape) >= 2:
                    selected_key = k
                    break
        
        if selected_key:
            return hf[selected_key][()]
        else:
            raise ValueError(f"No suitable data found in {filepath}. Keys: {keys}")

def resize_image(img, target_size=(256, 256)):
    """Resize numpy image [H, W] to target_size."""
    # Convert to PIL
    # Assumes img is normalized [0, 1] float
    # Map to [0, 255] for PIL
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize(target_size, Image.BICUBIC)
    # Back to [0, 1] float
    return np.array(pil_img).astype(np.float32) / 255.0

def process_mri(src_paths, save_path, num_samples=500, target_size=(256, 256)):
    """
    Process M4Raw MRI data.
    """
    print(f"Processing MRI from {len(src_paths)} paths...")
    files = []
    for p in src_paths:
        found = glob.glob(os.path.join(p, '**', '*.h5'), recursive=True)
        print(f"  Found {len(found)} .h5 files in {p}")
        files.extend(found)
    
    if not files:
        print("No MRI files found.")
        return

    data_list = []
    count = 0
    np.random.shuffle(files)
    
    for f in tqdm(files):
        if count >= num_samples: break
        try:
            vol = read_h5_volume(f)
            
            if vol.ndim == 2:
                vol = vol[np.newaxis, ...]
            elif vol.ndim == 3:
                if vol.shape[0] > vol.shape[2]: 
                    vol = vol.transpose(2, 0, 1)
            
            vol = normalize(vol)
            
            for i in range(vol.shape[0]):
                slc = vol[i]
                
                # Resize if needed
                if target_size is not None and slc.shape != target_size:
                    slc = resize_image(slc, target_size)
                
                pair = np.stack([slc, slc]) # [2, H, W]
                data_list.append(pair)
                
            count += 1
        except Exception as e:
            pass

    if data_list:
        # No need to filter by shape anymore if we resized
        all_data = np.stack(data_list, axis=0) # [N, 2, H, W]
        all_data = all_data.transpose(1, 0, 2, 3) # [2, N, H, W]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, all_data)
        print(f"Saved MRI appendix to {save_path} with shape {all_data.shape}")
    else:
        print("No MRI data processed.")

def process_busi(src_path, save_path, num_samples=500, target_size=(256, 256)):
    """
    Process BUSI Ultrasound data.
    Structure: src_path/{benign, malignant, normal}/...
    """
    print(f"Processing BUSI from {src_path}...")
    
    # Search for png files, excluding masks
    # Pattern: src_path/**/*.png
    all_files = glob.glob(os.path.join(src_path, '**', '*.png'), recursive=True)
    img_files = [f for f in all_files if '_mask' not in f]
    
    print(f"  Found {len(img_files)} image files (excluding masks).")
    
    data_list = []
    np.random.shuffle(img_files)
    
    for img_f in tqdm(img_files):
        try:
            # Read image
            # Convert to grayscale if needed (BUSI is usually grayscale but saved as RGB png)
            pil_img = Image.open(img_f).convert('L')
            img = np.array(pil_img).astype(np.float32)
            
            # Normalize
            img = normalize(img)
            
            # Resize
            if target_size is not None:
                img = resize_image(img, target_size)
            
            # Create pair [Low, Full]
            # Since we only have real ultrasound (noisy), we duplicate it.
            # Training loop should handle this (e.g. Noise2Noise or unsupervised)
            pair = np.stack([img, img]) # [2, H, W]
            data_list.append(pair)
            
        except Exception as e:
            print(f"Error processing {img_f}: {e}")

    if data_list:
        all_data = np.stack(data_list, axis=0) # [N, 2, H, W]
        all_data = all_data.transpose(1, 0, 2, 3) # [2, N, H, W]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, all_data)
        print(f"Saved BUSI appendix to {save_path} with shape {all_data.shape}")
    else:
        print("No BUSI data processed.")

if __name__ == "__main__":
    # MRI Paths
    mri_roots = [
        r"Z:\dataset\M4Raw_multicoil_test",
        r"Z:\dataset\M4RawV1.5_gre_data",
        r"Z:\dataset\M4RawV1.5_motion",
        r"Z:\dataset\M4RawV1.5_multicoil_train",
        r"Z:\dataset\M4RawV1.5_multicoil_val"
    ]
    
    # BUSI Path
    busi_root = r"Z:\dataset\Dataset_BUSI_with_GT"
    
    out_dir = r"d:\PycharmProjects\pythonProject\Pao_Lab\denoise\final\DU-GAN\dataset\cmayo_appendix"
    
    # Process MRI
    process_mri(mri_roots, os.path.join(out_dir, "mri_appendix.npy"))
    
    # Process BUSI
    process_busi(busi_root, os.path.join(out_dir, "busi_appendix.npy"))
