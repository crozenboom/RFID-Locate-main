#!/usr/bin/env python
import socket
import requests
import sys
import signal
import time

# Webhook URL
WEBHOOK_URL = "https://webhook.site/TagData"

# Reader details
READER_IP = "169.254.1.1"
READER_PORT = 5084

# Global flag and socket
running = True
sock = None

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down."""
    global running
    print("\nReceived shutdown signal (Ctrl+C).")
    running = False

def send_to_webhook(data):
    """Send raw data to the webhook via HTTP POST."""
    print(f"Preparing to send {len(data)} bytes to webhook: {data[:50]!r}...")
    try:
        response = requests.post(WEBHOOK_URL, data=data, headers={"Content-Type": "text/xml"})
        if response.status_code == 200:
            print(f"Successfully sent {len(data)} bytes to webhook. Response: {response.status_code}")
        else:
            print(f"Webhook returned error: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Failed to send to webhook: {e}")

def main():
    global running, sock

    signal.signal(signal.SIGINT, signal_handler)
    print("Signal handler set up for Ctrl+C.")

    print("Creating TCP socket...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    print(f"Socket created with 10-second connection timeout.")

    print(f"Attempting to connect to {READER_IP}:{READER_PORT}...")
    try:
        sock.connect((READER_IP, READER_PORT))
        print("Successfully connected to reader.")
    except socket.error as e:
        print(f"Connection failed: {e}")
        sock.close()
        sys.exit(-1)

    sock.settimeout(1.0)
    print("Set socket receive timeout to 1 second.")

    buffer = bytearray()
    last_send_time = time.time()
    print("Starting main loop to receive data...")
    while running:
        try:
            print("Waiting for data from reader...")
            data = sock.recv(4096)
            if not data:
                print("No data received; connection closed by reader.")
                break

            print(f"Received {len(data)} bytes from reader: {data[:50]!r}...")
            buffer.extend(data)

            # Process complete messages
            while b'</LLRPMessage>' in buffer:
                print("Found complete LLRP message in buffer.")
                end_idx = buffer.index(b'</LLRPMessage>') + len(b'</LLRPMessage>')
                message = buffer[:end_idx]
                print(f"Extracted message of {len(message)} bytes.")
                buffer = buffer[end_idx:]
                print(f"Remaining buffer size: {len(buffer)} bytes.")
                send_to_webhook(message)
                last_send_time = time.time()

            # Fallback: Send partial data if buffer grows or times out
            if buffer and (time.time() - last_send_time > 5.0):  # 5-second threshold
                print("No complete message in 5s; sending partial buffer.")
                send_to_webhook(buffer)
                buffer.clear()
                last_send_time = time.time()

        except socket.timeout:
            print("Socket timeout (1s); no data received, continuing...")
            if buffer and (time.time() - last_send_time > 5.0):
                print("No complete message in 5s; sending partial buffer.")
                send_to_webhook(buffer)
                buffer.clear()
                last_send_time = time.time()
            continue
        except socket.error as e:
            print(f"Socket error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    print("Exiting main loop; starting cleanup...")
    if sock:
        print("Closing socket...")
        sock.close()
        sock = None  # Prevent double-close
    print("Disconnected from reader.")

if __name__ == "__main__":
    main()