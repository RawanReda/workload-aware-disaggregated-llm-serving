import csv
import os
import subprocess
import threading
import time


class GPUMonitor:
    def __init__(self, csv_file="gpu_metrics.csv", interval=1):
        self.csv_file = csv_file
        self.interval = interval
        self.measurements = []
        self.stop_event = threading.Event()
        self.thread = None

    def query_gpu(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            for line in result.stdout.strip().splitlines():
                gpu_index, gpu_utilization, memory_used, memory_total = map(int, line.split(","))
                timestamp = time.time()

                measurement = {
                    "timestamp": timestamp,
                    "gpu_index": gpu_index,
                    "gpu_utilization": gpu_utilization,
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                }
                self.measurements.append(measurement)
                writer.writerow([timestamp, gpu_index, gpu_utilization, memory_used, memory_total])

    def _monitor(self):
        while not self.stop_event.is_set():
            self.query_gpu()
            self.stop_event.wait(self.interval)

    def start(self, csv_file=None):
        self.measurements = []
        self.stop_event.clear()

        if csv_file is not None:
            self.csv_file = csv_file

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "gpu_index", "gpu_utilization", "memory_used", "memory_total"])

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
        return self.measurements