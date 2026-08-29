import subprocess
import os
import sys

request_rates = [8, 16, 32]
workload_profile = [(1800, 100), (100, 1800), (950, 950)]


sub_folder_path = sys.argv[1]

results_dir = f"results/{sub_folder_path}"
os.makedirs(results_dir, exist_ok=True)

print(f"Running benchmarks for sub folder: {sub_folder_path}")

def run_benchmarks():
    for rate in request_rates:
        for input_len, output_len in workload_profile:
            bench_results_file = f"{results_dir}/rate_{rate}_input_{input_len}_output_{output_len}.txt"
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


def main():
    run_benchmarks()


if __name__ == "__main__":
    main()
