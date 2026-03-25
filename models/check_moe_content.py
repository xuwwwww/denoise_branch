import os

file_path = 'models/DUGAN/DUGAN_MoE.py'

if os.path.exists(file_path):
    print(f"--- Content of {file_path} ---")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'NoiseDiscriminator(' in line or 'input_nc=' in line or 'in_channels=' in line:
                print(f"{i+1}: {line.strip()}")
else:
    print(f"File {file_path} not found!")
