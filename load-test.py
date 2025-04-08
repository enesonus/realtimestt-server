import aiohttp
import asyncio
import time
import argparse

async def send_request(session, url, file_content, filename, form_fields):
    """
    Sends a single POST request with the given file and form fields.
    Measures and returns the client latency of the request along with the server's reported latency.
    """
    # Create a FormData instance for multipart/form-data upload.
    data = aiohttp.FormData()
    data.add_field('file', file_content, filename=filename, content_type='audio/wav')
    for key, value in form_fields.items():
        data.add_field(key, value)

    headers = {"ngrok-skip-browser-warning": "true"}

    start = time.monotonic()
    async with session.post(url, data=data, headers=headers) as response:
        # Parse the response as JSON to extract server_latency.
        resp_json = await response.json()
        print(f"Response: {resp_json}")
    client_latency = time.monotonic() - start

    # Extract the server latency from the response JSON.
    server_latency = resp_json.get("server_latency", None)
    return client_latency, server_latency

async def main(args):
    # Define the target URL and form fields.
    url = args.url
    filename = args.filename
    form_fields = {
        'temperature': "0.0",
        'temperature_inc': "0.2",
        'response_format': "json"
    }
    
    # Read the file content once for reuse.
    with open(filename, 'rb') as f:
        file_content = f.read()

    async with aiohttp.ClientSession() as session:
        # Create N concurrent tasks.
        tasks = [
            send_request(session, url, file_content, filename, form_fields)
            for _ in range(args.num_requests)
        ]
        # Run all tasks concurrently.
        results = await asyncio.gather(*tasks)

    # Separate the client latencies and server latencies.
    client_latencies = [result[0] for result in results]
    # Only include valid server latencies (non-None values) and convert them to float.
    server_latencies = [float(result[1]) for result in results if result[1] is not None]

    # Calculate and print client-side latency statistics.
    max_client_latency = max(client_latencies)
    min_client_latency = min(client_latencies)
    avg_client_latency = sum(client_latencies) / len(client_latencies)
    
    print(f"\nNumber of requests: {args.num_requests}")
    print(f"Client-side latency:")
    print(f"  Max: {max_client_latency * 1000:.2f} ms")
    print(f"  Min: {min_client_latency * 1000:.2f} ms")
    print(f"  Average: {avg_client_latency * 1000:.2f} ms")

    # Calculate and print server-side latency statistics if available.
    if server_latencies:
        max_server_latency = max(server_latencies)
        min_server_latency = min(server_latencies)
        avg_server_latency = sum(server_latencies) / len(server_latencies)
        print(f"\nServer-side latency:")
        print(f"  Max: {max_server_latency * 1000:.2f} ms")
        print(f"  Min: {min_server_latency * 1000:.2f} ms")
        print(f"  Average: {avg_server_latency * 1000:.2f} ms")
    else:
        print("\nServer latency not found in any of the responses.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load test the Whisper server inference endpoint")
    parser.add_argument('--url', type=str, default="https://1fb3-34-58-114-21.ngrok-free.app/v1/audio/transcriptions",
                        help="Inference endpoint URL")
    parser.add_argument('--filename', type=str, default="./rtts-test.wav",
                        help="Path to the WAV file to upload")
    parser.add_argument('--num_requests', type=int, default=15,
                        help="Number of concurrent requests to send")
    args = parser.parse_args()

    asyncio.run(main(args))
