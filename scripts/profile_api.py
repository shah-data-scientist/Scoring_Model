"""
API Performance Profiling Script

Measures latency and throughput of the Credit Scoring API.
Ensure the API is running before executing this script.
"""
import requests
import time
import numpy as np
import concurrent.futures
import statistics

API_URL = "http://localhost:8000/predict"
N_REQUESTS = 100
CONCURRENCY = 10

# Mock Data (matches EXPECTED_FEATURES=194)
mock_features = np.random.rand(194).tolist()
payload = {
    "features": mock_features,
    "client_id": "PERF_TEST"
}

def make_request(_):
    start = time.time()
    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        latency = (time.time() - start) * 1000  # ms
        return latency, resp.status_code
    except Exception as e:
        return None, str(e)

def profile_api():
    print(f"Profiling API at {API_URL}")
    print(f"Requests: {N_REQUESTS}, Concurrency: {CONCURRENCY}")
    
    latencies = []
    errors = 0
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(executor.map(make_request, range(N_REQUESTS)))
        
    total_time = time.time() - start_total
    
    for latency, status in results:
        if latency is not None:
            latencies.append(latency)
        else:
            errors += 1
            
    if not latencies:
        print("All requests failed. Is the API running?")
        return

    print("\nResults:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {N_REQUESTS / total_time:.2f} req/s")
    print(f"  Errors: {errors}")
    print(f"  Avg Latency: {statistics.mean(latencies):.2f} ms")
    print(f"  P95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 Latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min Latency: {min(latencies):.2f} ms")
    print(f"  Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    profile_api()
