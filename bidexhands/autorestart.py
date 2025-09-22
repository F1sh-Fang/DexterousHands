import os
import subprocess
import glob
import time
import yaml


def find_latest_checkpoint(path):
    list_of_files = glob.glob(os.path.join(path, "model_*.pt"))
    if not list_of_files:
        return None
    try:
        latest_file = max(list_of_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        return latest_file
    except (ValueError, IndexError):
        return None

if __name__ == '__main__':
    project_id = "Dof9_ik_0"
    log_path = "logs/BotyardHandPick/ppo/" + project_id
    first = True
    while True:

        latest_chkp = find_latest_checkpoint(log_path)

        base_cmd = ['python', 'train_botyard.py']
        if latest_chkp:
            print(f"--- Found latest checkpoint: {latest_chkp} ---")
            cmd = base_cmd + ['--model_dir', latest_chkp]
            if not first:
                cmd += ['--headless']
        else:
            print("--- No checkpoint found, starting training from scratch. ---")
            cmd = base_cmd

        try:
            print(f"--- Starting subprocess with command: {' '.join(cmd)} ---")
            result = subprocess.run(cmd, check=False) 
            first = False

            if result.returncode == 0:
                print("--- Training completed successfully. Exiting restart loop. ---")
                break  
            else:
                print(f"--- Subprocess terminated with return code: {result.returncode}. Restarting... ---")

        except KeyboardInterrupt:
            print("\n--- KeyboardInterrupt received. Shutting down autorestart script. ---")
            break
        except Exception as e:
            print(f"--- An unexpected error occurred: {e}. Restarting... ---")

        print("--- Waiting for 2 seconds before restarting... ---")
        time.sleep(2)