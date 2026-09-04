import os
import signal
import subprocess
import time

import torch
import yaml

with open("gpu_config.yaml", "r") as f:
    gpu_config = yaml.safe_load(f)

gpu_count = torch.cuda.device_count()
models = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"]
configs = gpu_config["gpu_configurations"].get(gpu_count)

if configs is None:
    raise ValueError(f"No GPU configuration found for {gpu_count} GPUs in gpu_config.yaml")


def normalize_stage_configs(stage_config):
    gpu_ids = stage_config["gpus"]
    if "tp" in stage_config and "pp" in stage_config:
        return [("", {"gpus": gpu_ids, "tp": stage_config["tp"], "pp": stage_config["pp"]})]

    normalized_configs = []
    for parallelism_configuration_name, parallelism_configuration in stage_config.items():
        if parallelism_configuration_name == "gpus":
            continue
        if (
            not isinstance(parallelism_configuration, dict)
            or "tp" not in parallelism_configuration
            or "pp" not in parallelism_configuration
        ):
            raise ValueError(
                f"Invalid parallelism configuration '{parallelism_configuration_name}': "
                "expected 'tp' and 'pp' values"
            )
        normalized_configs.append(
            (
                parallelism_configuration_name,
                {
                    "gpus": gpu_ids,
                    "tp": parallelism_configuration["tp"],
                    "pp": parallelism_configuration["pp"],
                },
            )
        )
    return normalized_configs


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



for model in models:
    print(f"Running experiments for model: {model}")
    for gpu_split, config in configs.items():
        prefill_configurations = normalize_stage_configs(config["prefill"])
        decode_configurations = normalize_stage_configs(config["decode"])

        for prefill_parallelism, prefill_config in prefill_configurations:
            for decode_parallelism, decode_config in decode_configurations:
                # Example: gpu_split_equal_prefill_tp_heavy_tp4_pp1_decode_pp_heavy_tp1_pp4
                configuration_name = "_".join(
                    filter(
                        None,
                        [
                            f"gpu_split_{gpu_split}",
                            (
                                f"prefill_{prefill_parallelism}_tp{prefill_config['tp']}_pp{prefill_config['pp']}"
                                if prefill_parallelism
                                else f"prefill_tp{prefill_config['tp']}_pp{prefill_config['pp']}"
                            ),
                            (
                                f"decode_{decode_parallelism}_tp{decode_config['tp']}_pp{decode_config['pp']}"
                                if decode_parallelism
                                else f"decode_tp{decode_config['tp']}_pp{decode_config['pp']}"
                            ),
                        ],
                    )
                )
                print(f"Running configuration: {configuration_name}")
                print(f"Prefill configuration: {prefill_config}")
                print(f"Decode configuration: {decode_config}")

                log_file = f"run_script_{model.split('/')[-1]}_{configuration_name}.txt"
                process = None

                try:
                    prefill_gpus = ",".join(map(str, prefill_config["gpus"]))
                    decoder_gpus = ",".join(map(str, decode_config["gpus"]))

                    if not prefill_gpus or not decoder_gpus:
                        print("Warning: empty GPU list for prefill or decoder; skipping this configuration.")
                        continue

                    process = subprocess.Popen(
                        [
                            "./disaggregated_prefill_example.sh",
                            prefill_gpus,
                            str(prefill_config["tp"]),
                            str(prefill_config["pp"]),
                            decoder_gpus,
                            str(decode_config["tp"]),
                            str(decode_config["pp"]),
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

                    gpu_split_name = f"p_{'_'.join(map(str, prefill_config['gpus']))}_d_{'_'.join(map(str, decode_config['gpus']))}"
                    model_folder = model.split("/")[-1]
                    results_dir = f"{model_folder}/gpu_split_{gpu_split_name}_{configuration_name}"

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
                    print(f"Warning: benchmark run failed for model={model}, configuration={configuration_name}: {exc}")
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

                    cleanup_ports([8000, 8100, 8200])
                    print("Experiment cleanup complete")
            
            