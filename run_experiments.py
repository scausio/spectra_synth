"""
Script to run multiple experiments with different model architectures
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main
from config_experiments import get_experiment_config, get_all_experiments, print_experiment_info
import json


def run_experiment(experiment_name):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Get configuration
    config = get_experiment_config(experiment_name)
    
    # Save config
    os.makedirs(config['outdir'], exist_ok=True)
    with open(os.path.join(config['outdir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training with config passed as argument
    try:
        main(config)
        print(f"\n✓ Experiment {experiment_name} completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Experiment {experiment_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_experiments():
    """Run all available experiments"""
    experiments = get_all_experiments()
    results = {}
    
    print(f"\nRunning {len(experiments)} experiments...\n")
    
    for exp_name in experiments:
        success = run_experiment(exp_name)
        results[exp_name] = 'Success' if success else 'Failed'
    
    # Print summary
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")
    for exp_name, status in results.items():
        status_symbol = "✓" if status == 'Success' else "✗"
        print(f"{status_symbol} {exp_name:30s}: {status}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model training experiments')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Name of experiment to run (from config_experiments.py)')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print_experiment_info()
    elif args.all:
        run_all_experiments()
    elif args.experiment:
        run_experiment(args.experiment)
    else:
        print("Please specify --experiment <name>, --all, or --list")
        print("\nAvailable experiments:")
        for exp in get_all_experiments():
            print(f"  - {exp}")
