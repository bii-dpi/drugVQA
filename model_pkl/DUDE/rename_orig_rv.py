import os
import shutil

for fname in os.listdir():
    if fname.startswith("orig_rv"):
        shutil.move(fname, f"123456789_{fname}")
