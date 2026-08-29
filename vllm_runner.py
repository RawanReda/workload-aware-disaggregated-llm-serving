import subprocess
import yaml
import torch
import time

with open("gpu_config.yaml", "r") as f:
    gpu_config = yaml.safe_load(f)

gpu_count = torch.cuda.device_count()

configs = gpu_config["gpu_configurations"][gpu_count]

for gpu_split, config in configs.items():
    print(f"Running vllm_runner.py with GPU split: {gpu_split}")
    prefill = config["prefill"]
    decoder = config["decode"]

    print(f"Prefill configuration: {prefill}")
    print(f"Decode configuration: {decoder}")

    log_file = "run_script.txt"

    # install vllm, nixl 
    process = subprocess.Popen( [ 
        "./disaggregated_prefill_example.sh", 
        ",".join(map(str, prefill["gpus"])), 
        str(prefill["tp"]),
        str(prefill["pp"]),
        ",".join(map(str, decoder["gpus"])),
        str(decoder["tp"]),
        str(decoder["pp"]),
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,)

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
        

    subprocess.run(["python3", "experimental_setup.py"])

    process.terminate()
    process.wait()
