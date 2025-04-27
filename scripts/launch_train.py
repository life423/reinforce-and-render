import argparse, subprocess, sys, os
from pathlib import Path

parser = argparse.ArgumentParser(description="Launch PPO training.")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# Call the actual training loop
cmd = [sys.executable, "-m", "ai_platform_trainer.agents.training_loop"]
if args.headless:
    cmd.append("--headless")
# Pop open a new terminal window on Windows for live mode
if os.name == "nt" and not args.headless:
    subprocess.Popen(["start", "cmd", "/k"] + cmd, shell=True)
else:
    subprocess.Popen(cmd)
