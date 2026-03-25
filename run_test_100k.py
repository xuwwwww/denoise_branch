import os

# Configuration
# Format: (Run Name, Checkpoint Iteration)
models_to_test = [
    ("1128_Original", 100000),
    ("1205_hybrid", 100000),
    ("1201_cbam_v1", 100000),
    ("1125_Frequency_Attention", 100000)
]

dataset_name = "combined_train"

print("Commands to test 100k models on Combined Dataset:")
print("-" * 50)

for run_name, iter_num in models_to_test:
    cmd = (
        f"python test.py "
        f"--model_name DUGAN "
        f"--run_name {run_name} "
        f"--test_dataset_name {dataset_name} "
        f"--resume_iter {iter_num} "
        f"--num_workers 4 "
        f"--batch_size 1"
    )
    print(cmd)
    print("-" * 50)
