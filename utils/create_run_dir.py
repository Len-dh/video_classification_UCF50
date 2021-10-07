import os 


def create_run_dir(run_dir):    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)