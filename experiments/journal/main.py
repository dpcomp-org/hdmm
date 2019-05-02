import sys
import argparse
import itertools
import subprocess

def run_sequentially(commands):
    for cmd in commands:
        print('Running command:', cmd)
        subprocess.run(cmd, shell=True)

def run_with_slurm(commands, sbatch):
    opts = ' '.join(['--%s %s' % (k,v) for k,v in sbatch.items()])
    #slurm = 'sbatch --job-name %s -n 1 --cpu_per_task 2 --time 30'
    for cmd in commands:
        cmd = 'sbatch %s --wrap "%s"' % (opts, cmd)
        subprocess.run(cmd, shell=True) 

def generate_all_commands(base, args):
    """
    :param base: the command to run the python file
    :param args: the configurations to use
        keys are arguments
        values are list possible settings
    
    Note: runs cartesian product of commands implied by args
    """
    keys = args.keys()
    vals = args.values()
    commands = []
    for config in itertools.product(*vals):
        opts = [' --%s %s' % (k,v) for k,v in zip(keys, config)]
        commands.append( base + ''.join(opts) )

    return commands

def get_experiments():
    trials = 15
    args = {}
    args['dataset'] = ['census', 'adult', 'loans']
    args['workload'] = [1,2]
    args['approx'] = [False, True]

    commands = generate_all_commands('python compute_strategies.py', args)

    sbatch = {}
    sbatch['job-name'] = 'hdmm'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 4
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '3:00:00'

    return trials*commands, sbatch

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--slurm', action='store_true', help='run commands on slurm')

    args = parser.parse_args()

    commands, sbatch = get_experiments()

    if args.slurm:
        run_with_slurm(commands, sbatch)
    else:
        run_sequentially(commands)   
 
