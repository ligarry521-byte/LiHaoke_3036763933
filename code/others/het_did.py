# -*- coding: utf-8 -*-
"""
run_all.py: Master script to execute the entire Card & Krueger 1994 replication pipeline
Order: data cleaning → Table2 → Table3&4 → Figure1 → Heterogeneous DID
This will generate all cleaned datasets, markdown tables and figures in output folder.
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_path: str):
    print("\n" + "="*60)
    print(f"▶ Running: {script_path}")
    print("="*60)
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_path} FAILED with code {result.returncode}")
    print(f"✅ Finished: {script_path}\n")

def main():
    # define scripts in execution order (dependency order)
    script_list = [
        "code/clean_data.py",
        "code/table2.py",
        "code/table3_table4.py",
        "code/figure1.py",
        "code/het_did.py"
    ]

    # check each file exists
    for s in script_list:
        if not Path(s).exists():
            raise FileNotFoundError(f"Missing script file: {s}")

    print("🚀 Starting full replication pipeline for Card & Krueger (1994)")
    for script in script_list:
        run_script(script)

    print("\n🎉 ALL SCRIPTS FINISHED SUCCESSFULLY!")
    print("Generated outputs are located in ./data/processed and ./output")

if __name__ == "__main__":
    main()
