import os
import sys

print(f"CWD: {os.getcwd()}")

print("\n--- dugan_utils Directory ---")
try:
    if os.path.exists('dugan_utils'):
        print(os.listdir('dugan_utils'))
    else:
        print("dugan_utils directory NOT FOUND")
except Exception as e:
    print(f"Could not list dugan_utils: {e}")

print("\n--- Testing dugan_utils import ---")
try:
    import dugan_utils
    print(f"SUCCESS: import dugan_utils ({dugan_utils.__file__})")
except ImportError as e:
    print(f"FAIL: import dugan_utils - {e}")

try:
    import dugan_utils.dataset
    print(f"SUCCESS: import dugan_utils.dataset ({dugan_utils.dataset.__file__})")
except ImportError as e:
    print(f"FAIL: import dugan_utils.dataset - {e}")
