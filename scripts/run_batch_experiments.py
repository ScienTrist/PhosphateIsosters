
import os
import sys
import json
import time
import subprocess

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

def run_experiment(config):
    """
    Executes a single pipeline run with a specific configuration.
    We import main inside the loop to ensure settings are refreshed if we were to modify them,
    but better yet, we can modify the main() function to accept parameters.
    """
    print("\n" + "#"*80)
    print(f" STARTING EXPERIMENT: {config['name']}")
    print(f" Settings: {config}")
    print("#"*80 + "\n")

    # ADVANCED: Instead of separate process, we can import and call a modified main
    from main_batch import main_run
    
    try:
        start_time = time.time()
        main_run(config)
        elapsed = time.time() - start_time
        print(f"\n[Finished] {config['name']} in {elapsed/3600:.2f} hours.")
    except Exception as e:
        import traceback
        print(f"\n[FAILED] {config['name']}: {e}")
        traceback.print_exc()

def main():
    # DEFINE YOUR OVERNIGHT QUEUE HERE
    experiments = [
        {
            "name": "Global_Search_99",
            "SEARCH_COV": 0.8,
            "SEARCH_IDENTITY": 0.5,
            "DB_CLUSTER_ID": 0.99,
            "TM_FAST_MODE": True
        },
        {
            "name": "Local_Search_99",
            "SEARCH_COV": 0.0,
            "SEARCH_IDENTITY": 0.5,
            "DB_CLUSTER_ID": 0.99,
            "TM_FAST_MODE": True
        },
        {
            "name": "Wide_Search_0.3_ID",
            "SEARCH_COV": 0.0,
            "SEARCH_IDENTITY": 0.3,
            "DB_CLUSTER_ID": 0.99,
            "TM_FAST_MODE": True
        }
    ]

    print(f"Queueing {len(experiments)} experiments for batch execution...")
    
    for i, config in enumerate(experiments):
        print(f"\nProcessing Job {i+1}/{len(experiments)}...")
        run_experiment(config)

    print("\n" + "="*80)
    print(" ALL BATCH JOBS COMPLETE ")
    print("="*80)

if __name__ == "__main__":
    main()
