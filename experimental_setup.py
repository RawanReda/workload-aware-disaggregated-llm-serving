import subprocess

request_rates = [1, 2, 4, 8, 16, 32]
workload_profile = [(1800, 100), (100, 1800), (950, 950)]


bench_results_file = "bench_results.txt"

for rate in request_rates:
    with open(bench_results_file, "a") as f:
        f.write(f"Request Rate: {rate}\n")
        f.write(f"Input length: {input_len}, Output length: {output_len}\n")
        for input_len, output_len in workload_profile:
            subprocess.run([
                "vllm", "bench", "serve",
                "--dataset-name", "random",
                "--input-len", str(input_len),
                "--output-len", str(output_len),
                "--request-rate", str(rate),
            ], stdout=f, stderr=subprocess.STDOUT, text=True)