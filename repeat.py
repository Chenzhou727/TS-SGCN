import os
import subprocess
import sys

# ⭐ Key (prevents libgomp errors + stabilizes training)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def run_commands(commands, times):
    for i in range(times):
        print(f"\n===== Run {i+1} =====\n")
        for cmd in commands:
            print(f"Executing: {cmd}")
            result = subprocess.run(cmd,shell=True, stdout=sys.stdout, stderr=sys.stderr)
            if result.returncode != 0:
                print(f"Command failed: {cmd}")
                return


# All experiments (unified format)
para = [
    ("nied3s","nied3s"),
]

num=1

for p2,p3 in para:
   
    commands = [
        f"python -u train.py --dataset {p2} --model_path {p3}",
        f"python -u predict.py --dataset {p2} --model_path {p3}",
    ]

    run_commands(commands, num)
    