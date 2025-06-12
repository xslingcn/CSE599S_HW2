import os
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any
import argparse

class ExperimentRunner:
    def __init__(self):
        self.experiments = self._define_experiments()
        
    def _define_experiments(self) -> List[Dict[str, Any]]:
        """Define all experiments to run.
        """
        experiments = []
        
        # 1.5 Sanity check experiments
        sanity_params = {
            'n_layer': 1,
            'n_embd': 32,
            'n_head': 4,
            'max_steps': 1000,
            'log_interval': 100,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'seed': 42,
            'eval_points_per_decade': 16,
        }
        
        # Basic sanity check
        exp = {
            'name': 'sanity_check',
            'data_dir': 'data/sanity_check',
            'out_dir': 'out/sanity_check',
            **sanity_params
        }
        experiments.append(exp)
        
        # Sanity check with masking
        exp = {
            'name': 'sanity_check_masked',
            'data_dir': 'data/sanity_check',
            'out_dir': 'out/sanity_check_masked',
            'mask_first_n': 3,
            **sanity_params
        }
        experiments.append(exp)
        
        # Common settings
        common_params = {
            'max_steps': 100000,
            'learning_rate': 1e-3,
            'log_interval': 1000,
            'n_embd': 128,
            'n_head': 4,
            'eval_points_per_decade': 16,
        }
        
        seeds = [42, 123, 456]
        modulus = [97, 113]
        n_layers = [1, 2]
        
        # 2.2 Addition and Subtraction experiments
        for operation in ['add', 'subtract']:
            for mod in modulus:
                for n_layer in n_layers:
                    for seed in seeds:
                        exp = {
                            'name': f'{operation}_mod{mod}_layer{n_layer}_seed{seed}',
                            'data_dir': f'data/algorithmic/{operation}_mod{mod}',
                            'out_dir': f'out/{operation}_mod{mod}_layer{n_layer}_seed{seed}',
                            'seed': seed,
                            'n_layer': n_layer,
                            'batch_size': 64,
                            **common_params
                        }
                        experiments.append(exp)
        
        # 2.3 Grokking, batch 512
        divide_params = {
            **common_params,
            'weight_decay': 1.0,
            'beta1': 0.9,
            'beta2': 0.98,
        }

        exp = {
            'name': f'divide_mod97_layer2_seed42_batch512',
            'data_dir': 'data/algorithmic/divide_mod97',
            'out_dir': f'out/divide_mod97_layer2_seed42_batch512',
            'seed': 42,
            'n_layer': 2,
            'batch_size': 512,
            **divide_params
        }
        experiments.append(exp)
        
        # 2.4 Ablation study, different batch sizes
        batch_configs = [
            (64, 'batch64'),
            (128, 'batch128'),
            (256, 'batch256'),
            (512, 'batch512'),
        ]
        
        for batch_size, batch_name in batch_configs:
            for seed in seeds:
                exp = {
                    'name': f'divide_mod97_layer2_seed{seed}_{batch_name}',
                    'data_dir': 'data/algorithmic/divide_mod97',
                    'out_dir': f'out/divide_mod97_layer2_seed{seed}_{batch_name}',
                    'seed': seed,
                    'n_layer': 2,
                    'batch_size': batch_size,
                    **divide_params
                }
                experiments.append(exp)

        
        return experiments
    
    def run_experiment(self, exp: Dict[str, Any]) -> int:
        """Run a single experiment and log output."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*60}")
        
        os.makedirs(exp['out_dir'], exist_ok=True)
        log_file = os.path.join(exp['out_dir'], 'training.log')
        
        # Build command
        cmd = [
            'python', 'train.py',
            '--data_dir', exp['data_dir'],
            '--out_dir', exp['out_dir'],
            '--seed', str(exp['seed']),
            '--n_layer', str(exp['n_layer']),
            '--n_embd', str(exp['n_embd']),
            '--n_head', str(exp['n_head']),
            '--batch_size', str(exp['batch_size']),
            '--max_steps', str(exp['max_steps']),
            '--learning_rate', str(exp['learning_rate']),
            '--log_interval', str(exp['log_interval']),
        ]
        
        # Optional parameters
        if 'weight_decay' in exp:
            cmd.extend(['--weight_decay', str(exp['weight_decay'])])
        if 'beta1' in exp:
            cmd.extend(['--beta1', str(exp['beta1'])])
        if 'beta2' in exp:
            cmd.extend(['--beta2', str(exp['beta2'])])
        if 'mask_first_n' in exp:
            cmd.extend(['--mask_first_n', str(exp['mask_first_n'])])
        if 'eval_points_per_decade' in exp:
            cmd.extend(['--eval_points_per_decade', str(exp['eval_points_per_decade'])])
        
        # Run the experiment and capture output
        try:
            with open(log_file, 'w') as f:
                # Write experiment info
                f.write(f"Experiment: {exp['name']}\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*60}\n\n")
                f.flush()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                return_code = process.wait()
                
                f.write(f"\n{'='*60}\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return code: {return_code}\n")
                
            return return_code
            
        except Exception as e:
            print(f"Error running experiment {exp['name']}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"\nError: {e}\n")
            return -1
    
    def run_all(self, start_from: int = 0):
        """Run all experiments starting from the given index."""
        total = len(self.experiments)
        successful = 0
        failed = []
        
        print(f"Total experiments to run: {total - start_from}")
        
        for i in range(start_from, total):
            exp = self.experiments[i]
            print(f"\nExperiment {i+1}/{total}: {exp['name']}")
            
            return_code = self.run_experiment(exp)
            
            if return_code == 0:
                successful += 1
                print(f"✓ Experiment {exp['name']} completed successfully")
            else:
                failed.append(exp['name'])
                print(f"✗ Experiment {exp['name']} failed with code {return_code}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {total - start_from}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed experiments:")
            for name in failed:
                print(f"  - {name}")
        
        return len(failed) == 0


def main():    
    parser = argparse.ArgumentParser(description='Run all modular arithmetic experiments')
    parser.add_argument('--start-from', type=int, default=0, 
                        help='Start from experiment index (useful for resuming)')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments without running them')
    parser.add_argument('--single', type=str, default=None,
                        help='Run only the experiment with this name')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.list:
        print("All experiments:")
        for i, exp in enumerate(runner.experiments):
            print(f"{i:3d}: {exp['name']}")
        return
    
    if args.single:
        # Find and run single experiment
        for exp in runner.experiments:
            if exp['name'] == args.single:
                return_code = runner.run_experiment(exp)
                sys.exit(0 if return_code == 0 else 1)
        print(f"Experiment '{args.single}' not found")
        sys.exit(1)
    
    # Run all experiments
    success = runner.run_all(start_from=args.start_from)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 