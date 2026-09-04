import os
import signal
import subprocess
import time

import torch
import yaml

with open("gpu_config.yaml", "r") as f:
    gpu_config = yaml.safe_load(f)

gpu_count = torch.cuda.device_count()
configs = gpu_config["gpu_configurations"].get(gpu_count)

if configs is None:
    raise ValueError(f"No GPU configuration found for {gpu_count} GPUs in gpu_config.yaml")


def cleanup_ports(ports):
    for port in ports:
        result = subprocess.run(
            ["bash", "-c", f"fuser {port}/tcp 2>/dev/null || true"],
            capture_output=True,
            text=True,
        )

        pids = result.stdout.split()

        if pids:
            print(f"Port {port} is being used by PIDs: {pids}")

            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Killed PID {pid}")
                except ProcessLookupError:
                    pass
        else:
            print(f"Port {port} is free")


models = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"]

for model in models:
    print(f"Running experiments for model: {model}")
    for gpu_split, config in configs.items():
        print(f"Running vllm_runner.py with GPU split: {gpu_split}")
        prefill = config["prefill"]
        decoder = config["decode"]

        print(f"Prefill configuration: {prefill}")
        print(f"Decode configuration: {decoder}")

        log_file = "run_script.txt"
        process = None

        try:
            prefill_gpus = ",".join(map(str, prefill["gpus"]))
            decoder_gpus = ",".join(map(str, decoder["gpus"]))

            if not prefill_gpus or not decoder_gpus:
                print("Warning: empty GPU list for prefill or decoder; skipping this configuration.")
                continue

            process = subprocess.Popen(
                [
                    "./disaggregated_prefill_example.sh",
                    prefill_gpus,
                    str(prefill["tp"]),
                    str(prefill["pp"]),
                    decoder_gpus,
                    str(decoder["tp"]),
                    str(decoder["pp"]),
                    str(model),
                ],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            while True:
                with open(log_file, "r") as f:
                    log_contents = f.read()
                    if "address already in use" in log_contents.lower():
                        print("ERROR: Address already in use. Stopping experiment.")
                        raise RuntimeError("Address already in use")
                    if "SERVERS_READY" in log_contents:
                        print("Servers are ready. Running run_benchmarks.py...")
                        break
                    if process.poll() is not None:
                        print("Warning: process terminated unexpectedly before readiness. Check the log file for details.")
                        raise RuntimeError("Process terminated unexpectedly.")
                time.sleep(1)

            gpu_split_name = f"p_{'_'.join(map(str, prefill['gpus']))}_d_{'_'.join(map(str, decoder['gpus']))}"
            model_folder = model.split("/")[-1]
            results_dir = f"{model_folder}/gpu_split_{gpu_split_name}"

            subprocess.run(
                [
                    "python3",
                    "run_benchmarks.py",
                    results_dir,
                    prefill_gpus,
                    decoder_gpus,
                ],
                check=True,
            )

        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
            raise
        except Exception as exc:
            print(f"Warning: benchmark run failed for model={model}, gpu_split={gpu_split}: {exc}")
            raise
        finally:
            if process is not None:
                try:
                    pgid = os.getpgid(process.pid)
                    print(f"Killing process group {pgid}")
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    print("Process group already exited.")

                process.wait()

            # Clean up orphans
            cleanup_ports([8000, 8100, 8200])
            print("Experiment cleanup complete")
            
            