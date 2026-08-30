import os
import subprocess
import sys

import pandas as pd

from gpu_monitor import GPUMonitor

request_rates = [8, 16, 32]
workload_profile = [(1800, 100), (100, 1800), (950, 950)]

sub_folder_path = sys.argv[1]
prefill_gpus = sys.argv[2].split(",")
decoder_gpus = sys.argv[3].split(",")


results_dir = f"results/{sub_folder_path}"
os.makedirs(results_dir, exist_ok=True)

print(f"Running benchmarks for sub folder: {sub_folder_path}")

monitor = GPUMonitor(interval=1)



def _summarise_series(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "max": series.max(),
        "min": series.min(),
        "p90": series.quantile(0.9),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
    }


def summarise_gpu_measurements(measurements):
    df = pd.DataFrame(measurements)
    empty_summary = {"gpu_utilization": {}, "memory_used": {}}

    if df.empty:
        return {}, empty_summary, empty_summary

    summary_by_gpu = {}
    for gpu_index, group in df.groupby("gpu_index"):
        summary_by_gpu[gpu_index] = {
            "gpu_utilization": _summarise_series(group["gpu_utilization"]),
            "memory_used": _summarise_series(group["memory_used"]),
        }

    prefill_indices = df[df["gpu_index"].isin([int(gpu) for gpu in prefill_gpus])]
    decoder_indices = df[df["gpu_index"].isin([int(gpu) for gpu in decoder_gpus])]

    prefill_summary = {
        "gpu_utilization": _summarise_series(prefill_indices["gpu_utilization"]),
        "memory_used": _summarise_series(prefill_indices["memory_used"]),
    }

    decoder_summary = {
        "gpu_utilization": _summarise_series(decoder_indices["gpu_utilization"]),
        "memory_used": _summarise_series(decoder_indices["memory_used"]),
    }

    return summary_by_gpu, prefill_summary, decoder_summary

def run_benchmarks():
    for rate in request_rates:
        for input_len, output_len in workload_profile:
            bench_results_file = f"{results_dir}/rate_{rate}_input_{input_len}_output_{output_len}.txt"
            csv_file = f"{results_dir}/rate_{rate}_input_{input_len}_output_{output_len}.csv"
            monitor.start(csv_file)

            with open(bench_results_file, "w") as f:
                f.write(f"Request Rate: {rate}\n")
                f.write(f"Input length: {input_len}, Output length: {output_len}\n")
                subprocess.run([
                    "vllm", "bench", "serve",
                    "--base-url", "http://127.0.0.1:8000",
                    "--dataset-name", "random",
                    "--input-len", str(input_len),
                    "--output-len", str(output_len),
                    "--request-rate", str(rate),
                    "--disable-tqdm",
                ], stdout=f, stderr=subprocess.STDOUT, text=True)

                f.write("===============================================================\n")
                gpu_measurements = monitor.stop()
                gpu_summary = summarise_gpu_measurements(gpu_measurements)

                f.write("\nGPU Measurements by GPU:\n")
                for gpu_index, summary in gpu_summary[0].items():
                    f.write(f"GPU {gpu_index} Summary: \n {summary}\n")

                f.write("\nGPU Summary:\n")
                f.write(f"Prefill Summary: \n {gpu_summary[1]}\n")
                f.write(f"Decoder Summary: \n {gpu_summary[2]}\n")

def main():
    run_benchmarks()


if __name__ == "__main__":
    main()
