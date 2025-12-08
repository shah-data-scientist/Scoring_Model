import requests
import numpy as np
import os

# Configuration for API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
EXPECTED_FEATURES = 194 # As defined in your API

def run_test_prediction():
    print(f"Attempting to send prediction request to: {API_URL}")

    # Generate random features (189 total)
    features = np.random.rand(EXPECTED_FEATURES).tolist()
    payload = {'features': features, 'client_id': 'TEST_CLIENT'}

    try:
        # Send POST request to the API
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        print("\n--- API Response ---")
        print(f"Status Code: {resp.status_code}")
        print(f"Response Body: {resp.json()}")
        print("--------------------")

    except requests.exceptions.ConnectionError:
        print(f'\nConnection Error: Could not connect to the API at {API_URL}.')
        print('Please ensure your FastAPI server is running in another terminal.')
    except requests.exceptions.Timeout:
        print(f'\nTimeout Error: The request to {API_URL} took too long to respond.')
    except requests.exceptions.RequestException as e:
        print(f'\nRequest Error: An unexpected error occurred during the request to {API_URL}.')
        print(f'Status Code: {resp.status_code if "resp" in locals() else "N/A"}')
        print(f'Response Body: {resp.text if "resp" in locals() else "N/A"}')
        print(f'Details: {e}')
    except Exception as e:
        print(f'\nAn unexpected Python error occurred: {e}')

if __name__ == "__main__":
    run_test_prediction()
