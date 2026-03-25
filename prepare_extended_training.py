import os
import shutil
import subprocess

# Configuration
# Format: (Source Run Name, New Run Name)
models_to_extend = [
    ("1128_Original", "1128_Original_Extended"),
    ("1205_hybrid", "1205_hybrid_Extended"),
    ("1201_cbam_v1", "1201_cbam_v1_Extended"),
    ("1125_Frequency_Attention", "1125_Frequency_Attention_Extended")
]

resume_iter = 100000
max_iter = 150000
dataset_name = "combined_train"
base_output_dir = "output"

def run_command(cmd):
    print(f"Running: {cmd}")
    # subprocess.run(cmd, shell=True, check=True) 
    # We will just print the commands for the user to run, or execute them if they want.
    # But running training in foreground sequentially might take forever.
    # Better to just prepare the folders and print the commands.
    pass

print("Preparing for Extended Training (100k -> 150k)...")

for src_name, new_name in models_to_extend:
    src_dir = os.path.join(base_output_dir, f"DUGAN_{src_name}", "save_models")
    new_dir = os.path.join(base_output_dir, f"DUGAN_{new_name}", "save_models")
    
    print(f"\n--- Processing {src_name} -> {new_name} ---")
    
    # 1. Create New Directory
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Created directory: {new_dir}")
    else:
        print(f"Directory exists: {new_dir}")
        
    # 2. Copy Checkpoints
    # We need to copy files ending with -100000
    # Since we are on server, we can't use local shutil easily if paths differ.
    # But this script is meant to run ON THE SERVER.
    
    if os.path.exists(src_dir):
        files = os.listdir(src_dir)
        copied_count = 0
        for f in files:
            if f.endswith(f"-{resume_iter}"):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(new_dir, f)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
        print(f"Copied {copied_count} checkpoint files from {src_dir}")
    else:
        print(f"Warning: Source directory {src_dir} not found!")

    # 3. Generate Training Command
    cmd = (
        f"python main.py "
        f"--model_name DUGAN "
        f"--run_name {new_name} "
        f"--train_dataset_name {dataset_name} "
        f"--resume_iter {resume_iter} "
        f"--max_iter {max_iter} "
        f"--batch_size 64 "
        f"--save_freq 2500 "
        f"--test_dataset_name cmayo_test_512 "
        f"--num_workers 4"
    )
    
    print("Training Command:")
    print(cmd)
    
print("\nDone! Please run the commands above sequentially or in parallel.")
