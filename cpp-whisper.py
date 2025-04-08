import subprocess
import time
from ngrok import ngrok  # Install via: pip install ngrok

def start_whisper_server():
    """
    Starts the whisper server process with the specified command.
    """
    command = [
        "/content/build/bin/whisper-server",
        "-m", "/content/whisper.cpp/models/ggml-large-v3-turbo.bin",
        "--port", "8011"
    ]
    process = subprocess.Popen(command)
    print("Whisper server started with PID:", process.pid)
    return process

def start_ngrok_tunnel():
    """
    Creates an ngrok tunnel for port 8011.
    Refer to https://pypi.org/project/ngrok/ for additional configuration details.
    """
    # If you haven't already, you can set your ngrok auth token:
    # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
    
    # Create an HTTP tunnel on port 8011.
    tunnel = ngrok.connect(8011, "http")
    print("ngrok tunnel established at:", tunnel.url())
    return tunnel

def main():
    # Start the whisper server.
    server_process = start_whisper_server()
    
    # Give the server time to start up.
    time.sleep(2)
    
    # Start the ngrok tunnel.
    tunnel = start_ngrok_tunnel()
    
    try:
        print("Server and tunnel are running. Press Ctrl+C to stop.")
        # Keep the main thread alive to maintain the tunnel.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown initiated.")
    finally:
        # Clean up: terminate the whisper server process.
        server_process.terminate()
        server_process.wait()
        # Disconnect the ngrok tunnel.
        ngrok.disconnect(tunnel.url())
        print("Server and tunnel have been shut down.")

if __name__ == "__main__":
    main()
