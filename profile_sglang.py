import time
import requests
import json
import random
import pandas as pd
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('../llama2')

file_path = './data.parquet'
df = pd.read_parquet(file_path)

texts = df['text'].tolist()

def _wait_and_warmup(url, headers):
    for _ in range(120):
        time.sleep(0.5)
        try:
            requests.get(url + "/get_model_info", timeout=5, headers=headers)
            break
        except requests.exceptions.RequestException:
            pass

    try:
        res = requests.post(
            url + "/generate",
            json={
                "text": "The capital city of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            headers=headers,
            timeout=600,
        )
        assert res.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Warmup request failed: {e}")

def flush_cache(url, headers):
    try:
        res = requests.get(url + "/flush_cache", headers=headers)
        assert res.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Cache flush request failed: {e}")

def generate_text_for_tokens(token_count, texts):
    combined_text = ""
    current_tokens = []

    while len(current_tokens) < token_count:
        if not texts:
            break
        text = texts.pop(0)
        combined_text += text + " "
        current_tokens = tokenizer.encode(combined_text, add_special_tokens=False)

    return tokenizer.decode(current_tokens[:token_count])

def measure_prefill_time(url, headers, batch_size, token_count, texts):
    texts_for_batch = [generate_text_for_tokens(token_count, texts[:]) for _ in range(batch_size)]
    print(texts_for_batch)

    print(f"{batch_size}")
    payload = {
        "text": texts_for_batch,
        "batch_size": batch_size,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
        },
    }

    start_time = time.time()
    try:
        res = requests.post(
            url + "/generate",
            json=payload,
            headers=headers,
            timeout=600,
        )
        assert res.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    end_time = time.time()
    return end_time - start_time

def main():
    headers = {}
    url = "http://127.0.0.1:30000"  # Replace with your server URL

    # _wait_and_warmup(url, headers)

    batch_sizes = [1,2,4,8,16,32,64]
    token_counts = [1,2,4,8,16,32,64,128,256,512,1024,2048]
# #
    # batch_sizes = [16]
    # token_counts = [512,1024]



    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}
        for token_count in token_counts:
            flush_cache(url, headers)
            time_taken = measure_prefill_time(url, headers, batch_size, token_count, texts)
            results[batch_size][token_count] = time_taken
            time.sleep(5)
            print(f"Batch size: {batch_size}, Token count: {token_count}, Time taken: {time_taken}")
            

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
