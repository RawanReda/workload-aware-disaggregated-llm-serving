import subprocess
import yaml
import torch
import time
import os 
import signal 

with open("gpu_config.yaml", "r") as f:
    gpu_config = yaml.safe_load(f)

gpu_count = torch.cuda.device_count()

configs = gpu_config["gpu_configurations"][gpu_count]

models = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B" ]


for model in models:
    print(f"Running experiments for model: {model}")
    for gpu_split, config in configs.items():
        print(f"Running vllm_runner.py with GPU split: {gpu_split}")
        prefill = config["prefill"]
        decoder = config["decode"]

        print(f"Prefill configuration: {prefill}")
        print(f"Decode configuration: {decoder}")

        log_file = "run_script.txt"

        try: 
            process = subprocess.Popen( [ 
                "./disaggregated_prefill_example.sh", 
                ",".join(map(str, prefill["gpus"])), 
                str(prefill["tp"]),
                str(prefill["pp"]),
                ",".join(map(str, decoder["gpus"])),
                str(decoder["tp"]),
                str(decoder["pp"]),
                str(model)
                ],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True)

            while True:
                with open(log_file, "r") as f:
                    log_contents = f.read()
                if "SERVERS_READY" in log_contents:
                    print("Servers are ready. Running experimental_setup.py...")
                    break
                if process.poll() is not None:
                    print("Process terminated unexpectedly. Check the log file for details.")
                    raise RuntimeError("Process terminated unexpectedly.")
                time.sleep(1)
        
            gpu_split = f"p_{'_'.join(map(str, prefill['gpus']))}_d_{'_'.join(map(str, decoder['gpus']))}"
            model_folder = model.split("/")[-1]
            results_dir = f"{model_folder}/gpu_split_{gpu_split}"
            subprocess.run(["python3", "experimental_setup.py", results_dir])
        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
        finally: 
            print("Force killing process group...")
            os.killpg(
                os.getpgid(process.pid),
                signal.SIGKILL
            )
            process.wait()
            print("Experiment cleanup complete")
