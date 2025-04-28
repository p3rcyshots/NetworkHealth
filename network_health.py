# -*- coding: utf-8 -*-
import psutil
import argparse
import time
import requests
import json
import sys
import os
import platform
import subprocess
import re
import threading
import queue
import socket
from collections import defaultdict

# --- Optional GPU Import ---
try:
    import pynvml
    pynvml_imported = True # Flag indicating import worked
except ImportError:
    pynvml_imported = False

# --- Configuration ---
DEFAULT_PING_TARGET = "8.8.8.8"
PING_COUNT = 4
DEFAULT_UPDATE_INTERVAL_SECONDS = 5
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# ANSI Color Codes
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# --- Helper Functions ---

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_color(text, color):
    print(f"{color}{text}{COLOR_RESET}")

def get_active_interfaces():
    # (Code from previous version - unchanged)
    active_interfaces = []
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        io_counters = psutil.net_io_counters(pernic=True)
    except Exception as e:
        print_color(f"Error accessing network interface information: {e}", COLOR_RED)
        return []

    for iface, iface_stats in stats.items():
        if iface not in addrs or iface not in io_counters: continue
        try:
            is_up = iface_stats.isup
            has_ip = any(addr.family in [socket.AF_INET, socket.AF_INET6] for addr in addrs[iface])
            is_loopback = any((addr.address.startswith("127.") and addr.family == socket.AF_INET) or \
                              (addr.address == "::1" and addr.family == socket.AF_INET6) \
                              for addr in addrs[iface])
            has_traffic = (io_counters[iface].bytes_sent > 0 or io_counters[iface].bytes_recv > 0)
            if is_up and has_ip and not is_loopback and has_traffic:
                 active_interfaces.append(iface)
        except Exception as e:
            print_color(f"Warning: Could not fully process interface {iface}: {e}", COLOR_YELLOW)
            continue

    if not active_interfaces:
         print_color("Warning: Could not detect active interfaces *with traffic*. Falling back to UP interfaces with IP (excluding loopback).", COLOR_YELLOW)
         fallback_interfaces = []
         for iface, iface_stats in stats.items():
             if iface not in addrs: continue
             try:
                is_up = iface_stats.isup
                has_ip = any(addr.family in [socket.AF_INET, socket.AF_INET6] for addr in addrs[iface])
                is_loopback = any((addr.address.startswith("127.") and addr.family == socket.AF_INET) or \
                                  (addr.address == "::1" and addr.family == socket.AF_INET6) \
                                  for addr in addrs[iface])
                if is_up and has_ip and not is_loopback:
                     fallback_interfaces.append(iface)
             except Exception as e:
                print_color(f"Warning: Could not fully process fallback interface {iface}: {e}", COLOR_YELLOW)
                continue
         if not fallback_interfaces:
              print_color("Error: No suitable network interfaces found at all.", COLOR_RED)
              return []
         else: return fallback_interfaces
    return active_interfaces

def run_ping(target, count, result_queue):
    # (Code from previous version - unchanged)
    avg_latency = None; packet_loss = None; command = []
    system = platform.system().lower()
    try:
        if system == "windows": command = ["ping", "-n", str(count), target]
        elif system == "darwin": command = ["ping", "-c", str(count), target]
        else: command = ["ping", "-c", str(count), target]
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=count * 1.5 + 2)
        output = process.stdout + process.stderr
        if system == "windows":
            loss_match = re.search(r"Lost = \d+ \((\d+)%\s?loss\)", output)
            avg_match = re.search(r"Average = (\d+)ms", output)
            if loss_match: packet_loss = float(loss_match.group(1))
            else:
                 if "Destination host unreachable" in output or "Request timed out" in output or "General failure" in output: packet_loss = 100.0
                 elif "Packets: Sent = " in output and "Received = 0" in output: packet_loss = 100.0
            if avg_match: avg_latency = float(avg_match.group(1))
            elif packet_loss is not None and packet_loss < 100:
                 latencies = re.findall(r"Reply from .*?: .*? time[=<](\d+)", output)
                 if latencies:
                     numeric_latencies = [float(l) if l != '<1' else 0.5 for l in latencies]
                     if numeric_latencies: avg_latency = sum(numeric_latencies) / len(numeric_latencies)
        else: # Linux / macOS
            loss_match = re.search(r"(\d+(\.\d+)?)\s*%\s*packet\s*loss", output)
            avg_match = re.search(r"(?:min/avg/max|round-trip min/avg/max)/(?:stddev|mdev)\s*=\s*.*?/([\d\.]+)/.*?\s*ms", output, re.IGNORECASE)
            if loss_match: packet_loss = float(loss_match.group(1))
            if avg_match: avg_latency = float(avg_match.group(1))
        if packet_loss is None:
             if "100% packet loss" in output or " 0 received" in output: packet_loss = 100.0
             elif " 0% packet loss" in output or f" {count} received" in output: packet_loss = 0.0
             else:
                print_color(f"Warning: Could not reliably parse packet loss. Output sample:\n{output[:250]}...", COLOR_YELLOW)
                packet_loss = None
        if avg_latency is None and (packet_loss is None or packet_loss < 100.0):
             print_color(f"Warning: Could not parse average latency. Output sample:\n{output[:250]}...", COLOR_YELLOW)
    except FileNotFoundError: print_color("Error: 'ping' command not found.", COLOR_RED); packet_loss = None
    except subprocess.TimeoutExpired: print_color(f"Error: Ping command to {target} timed out.", COLOR_RED); packet_loss = 100.0
    except PermissionError: print_color("Error: Permission denied running ping.", COLOR_RED); packet_loss = None
    except Exception as e: print_color(f"Error running or parsing ping: {e}", COLOR_RED); packet_loss = None
    result_queue.put({"latency": avg_latency, "packet_loss": packet_loss})

def call_ollama(prompt, model_name):
    # (Code from previous version - unchanged)
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        if 'response' in response_data: return response_data['response'].strip()
        elif 'error' in response_data: return f"Error from Ollama: {response_data['error']}"
        elif 'choices' in response_data and response_data['choices']: return response_data['choices'][0].get('text', '').strip()
        elif 'result' in response_data: return response_data['result'].strip()
        else: print_color(f"Warning: Unexpected Ollama response format: {response_data}", COLOR_YELLOW); return "Error: Could not parse Ollama response."
    except requests.exceptions.ConnectionError: return f"Error: Could not connect to Ollama API at {OLLAMA_API_URL}."
    except requests.exceptions.Timeout: return "Error: Request to Ollama timed out."
    except requests.exceptions.HTTPError as e:
        error_body = ""; status_code = e.response.status_code
        try: error_body = response.json().get('error', response.text)
        except json.JSONDecodeError: error_body = response.text
        return f"Error: Ollama API request failed ({status_code}). Message: {error_body}"
    except requests.exceptions.RequestException as e: return f"Error: Ollama request error: {e}"
    except json.JSONDecodeError: return f"Error: Could not decode Ollama JSON response: {response.text}"
    except Exception as e: return f"An unexpected error occurred calling Ollama: {e}"

# --- CORRECTED GPU Function ---
def get_gpu_stats():
    """
    Gets GPU utilization for NVIDIA GPUs using pynvml.
    Returns a list of dictionaries, one per GPU: [{'name': str, 'load': float}]
    Returns empty list if no GPUs found. Returns None if error during call.
    """
    # Assumes nvml is initialized if this function is called.
    gpu_stats = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return [] # Successful check, but no devices found

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): name = name.decode('utf-8') # Decode bytes to string
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_stats.append({
                'name': name,
                'load': float(utilization.gpu)
            })
        return gpu_stats
    except pynvml.NVMLError as error:
        # --- REMOVED global pynvml_available and assignment ---
        if error.args[0] == pynvml.NVML_ERROR_GPU_IS_LOST:
            print_color("NVML Error: GPU is lost or fallen off the bus.", COLOR_YELLOW)
        elif error.args[0] == pynvml.NVML_ERROR_NOT_SUPPORTED:
             print_color("NVML Error: Feature not supported by this GPU or driver.", COLOR_YELLOW)
        elif error.args[0] == pynvml.NVML_ERROR_UNKNOWN:
             print_color("An unknown NVML error occurred while getting stats.", COLOR_YELLOW)
        else: # Catch other NVML errors during stat fetching
            print_color(f"NVML Error ({error.args[0]}) during stat fetch: {error}", COLOR_YELLOW)
        return None # Indicate failure for this specific call
    except Exception as e:
        print_color(f"Unexpected error getting GPU stats: {e}", COLOR_YELLOW)
        return None # Indicate failure for this specific call

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Agentic AI Network/System Health Monitor using Ollama.")
    parser.add_argument("-m", "--model", required=True, help="Ollama model name")
    parser.add_argument("-p", "--ping-target", dest='target', default=DEFAULT_PING_TARGET, help=f"Ping target (default: {DEFAULT_PING_TARGET})")
    parser.add_argument("-t", "--time", dest='interval', type=float, default=DEFAULT_UPDATE_INTERVAL_SECONDS, help=f"Update interval (s) (default: {DEFAULT_UPDATE_INTERVAL_SECONDS})")
    args = parser.parse_args()

    if args.interval <= 0: print_color("Error: Update time interval must be positive.", COLOR_RED); sys.exit(1)

    # --- NVML Initialization ---
    nvml_initialized = False
    # Use the import flag, not the potentially modified global
    if pynvml_imported:
        try:
            print("Initializing NVML for GPU monitoring...")
            pynvml.nvmlInit()
            nvml_initialized = True # Set local flag for loop/shutdown control
            print_color("NVML initialized successfully.", COLOR_GREEN)
        except pynvml.NVMLError as error:
            print_color(f"Failed to initialize NVML: {error}. GPU monitoring disabled.", COLOR_YELLOW)
            # No need to modify global pynvml_available, nvml_initialized=False handles it
        except Exception as e:
             print_color(f"Unexpected error initializing NVML: {e}. GPU monitoring disabled.", COLOR_YELLOW)

    print(f"\nStarting Network & System Health Monitor...")
    print(f"Using Ollama Model: {args.model}")
    print(f"Ping Target (-p): {args.target}")
    print(f"Update Time (-t): {args.interval} seconds")
    print(f"GPU Monitoring: {'Enabled (NVIDIA)' if nvml_initialized else 'Disabled'}")
    print("Detecting active network interfaces...")

    last_net_stats = {}; active_interfaces = get_active_interfaces()
    if not active_interfaces:
        print_color("Exiting due to lack of suitable interfaces.", COLOR_RED)
        if nvml_initialized: pynvml.nvmlShutdown()
        sys.exit(1)

    print(f"Monitoring interfaces: {', '.join(active_interfaces)}")
    time.sleep(2)
    ping_results_queue = queue.Queue(); ping_thread = None
    last_ping_results = {"latency": None, "packet_loss": None}

    try:
        while True:
            clear_screen(); current_time = time.time()
            print(f"--- Network & System Health Monitor ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"--- Author: Prashant Saxena (https://github.com/p3rcyshots) ---")
            print(f"")
            print(f"Model: {args.model} | Ping: {args.target} | Update: {args.interval}s")
            print("-" * 60)

            # --- CPU Usage ---
            try: cpu_usage = psutil.cpu_percent(interval=None); print(f"CPU Usage: {cpu_usage:.1f}%")
            except Exception as e: print_color(f"Error CPU: {e}", COLOR_YELLOW); cpu_usage = -1.0

            # --- GPU Usage ---
            # Call get_gpu_stats ONLY if nvml was successfully initialized
            gpu_data = get_gpu_stats() if nvml_initialized else None
            print("\nGPU Status:")
            if nvml_initialized:
                if gpu_data is not None: # Check if the call succeeded
                    if gpu_data: # Check if list is not empty
                        for i, gpu in enumerate(gpu_data):
                            print_color(f"  GPU {i} ({gpu.get('name', '?')}): {gpu.get('load', 'N/A'):.1f}% Util", COLOR_BLUE)
                    else:
                        print("  No NVIDIA GPUs detected by NVML.")
                else:
                    # get_gpu_stats returned None, indicating an error during the call
                    print_color("  Error retrieving GPU stats this cycle.", COLOR_YELLOW)
            else:
                # NVML was not initialized successfully
                 print(f"  N/A ({'pynvml not installed' if not pynvml_imported else 'NVML init failed'})")


            # --- Network Bandwidth ---
            network_data = defaultdict(lambda: {'download_mbps': 0.0, 'upload_mbps': 0.0})
            active_interfaces_found_in_io = []
            try:
                current_net_io = psutil.net_io_counters(pernic=True)
                # (Bandwidth calculation logic - unchanged)
                for iface in active_interfaces:
                    if iface in current_net_io:
                        active_interfaces_found_in_io.append(iface)
                        stats = current_net_io[iface]
                        if iface in last_net_stats:
                            last_vals = last_net_stats[iface]; time_delta = current_time - last_vals['time']
                            if time_delta > 0.1:
                                recv_delta = stats.bytes_recv - last_vals['bytes_recv']; sent_delta = stats.bytes_sent - last_vals['bytes_sent']
                                if recv_delta < 0: recv_delta = stats.bytes_recv
                                if sent_delta < 0: sent_delta = stats.bytes_sent
                                recv_bps = recv_delta / time_delta; sent_bps = sent_delta / time_delta
                                network_data[iface]['download_mbps'] = max(0, (recv_bps * 8) / (1024 * 1024))
                                network_data[iface]['upload_mbps'] = max(0, (sent_bps * 8) / (1024 * 1024))
                        last_net_stats[iface] = {'bytes_sent': stats.bytes_sent, 'bytes_recv': stats.bytes_recv, 'time': current_time}

                if not active_interfaces_found_in_io and len(active_interfaces) > 0:
                     # (Interface re-detection logic - unchanged)
                     print_color("Warning: Monitored interfaces reported no I/O data. Re-detecting...", COLOR_YELLOW)
                     new_active_interfaces = get_active_interfaces()
                     if not new_active_interfaces: print_color("FATAL: Lost interfaces. Exiting.", COLOR_RED); sys.exit(1) # Add shutdown?
                     elif set(new_active_interfaces) != set(active_interfaces):
                         print_color(f"Interfaces changed: Now monitoring {', '.join(new_active_interfaces)}. Resetting stats.", COLOR_YELLOW)
                         active_interfaces = new_active_interfaces; last_net_stats = {}; network_data.clear()
            except Exception as e: print_color(f"Error NetIO: {e}", COLOR_YELLOW)

            print("\nNetwork Bandwidth:")
            # (Bandwidth display logic - unchanged)
            if not network_data and not last_net_stats: print(" (Waiting for first interval...)")
            elif not active_interfaces_found_in_io and last_net_stats: print(" (No I/O data reported this interval)")
            else:
                displayed_data = False
                for iface in active_interfaces:
                     if iface in network_data and iface in active_interfaces_found_in_io:
                         data = network_data[iface]; print_color(f"  {iface}:", COLOR_BLUE)
                         print(f"    Down: {data['download_mbps']:.2f} Mbps, Up: {data['upload_mbps']:.2f} Mbps"); displayed_data = True
                if not displayed_data and active_interfaces: print(" (No bandwidth calculated this interval)")

            # --- Ping Status ---
            if ping_thread is None or not ping_thread.is_alive():
                ping_thread = threading.Thread(target=run_ping, args=(args.target, PING_COUNT, ping_results_queue), daemon=True); ping_thread.start()
            ping_data_available = False
            try:
                while not ping_results_queue.empty(): last_ping_results = ping_results_queue.get_nowait(); ping_data_available = True
            except queue.Empty: pass
            print("\nPing Status:")
            print(f"  Target: {args.target}")
            latency = last_ping_results.get('latency'); loss = last_ping_results.get('packet_loss')
            latency_str = f"{latency:.2f} ms" if latency is not None else "N/A"
            loss_str = f"{loss:.1f}%" if loss is not None else "N/A"
            print(f"  Avg Latency: {latency_str}"); print(f"  Packet Loss: {loss_str}")
            if not ping_data_available and (ping_thread and ping_thread.is_alive()): print(f"  (Waiting for ping...)")

            print("-" * 60)

            # --- Prepare Prompt ---
            prompt = f"""You are an AI system health assistant. Analyze metrics below.
Identify potential problems: high bandwidth usage, latency >150ms, packet loss >2%, CPU >85%, GPU >80%.
Summarize concisely. Start with 'ALERT:' if problem found, else 'Status:'. Be brief.

Metrics (use N/A if unavailable):
- CPU Usage: {f"{cpu_usage:.1f}%" if cpu_usage >= 0 else "N/A"}
"""
            # Add GPU Data
            if nvml_initialized:
                if gpu_data is not None:
                    if gpu_data:
                        prompt += "- GPU Status:\n"
                        for i, gpu in enumerate(gpu_data):
                             load = gpu.get('load', -1.0)
                             prompt += f"  - GPU {i} ({gpu.get('name', '?')}): Util={f'{load:.1f}%' if load >=0 else 'N/A'}\n"
                    else: prompt += "- GPU Status: No NVIDIA GPUs detected\n"
                else: prompt += "- GPU Status: Error retrieving stats this cycle\n"
            else: prompt += f"- GPU Status: N/A ({'pynvml not installed' if not pynvml_imported else 'NVML init failed'})\n"

            # Add Network Data
            prompt += "- Network Interfaces:\n"
            interfaces_with_data = [iface for iface in active_interfaces if iface in network_data and iface in active_interfaces_found_in_io]
            if interfaces_with_data:
                for iface in interfaces_with_data:
                    data = network_data[iface]
                    prompt += f"  - {iface}: Down={data['download_mbps']:.2f} Mbps, Up={data['upload_mbps']:.2f} Mbps\n"
            elif not network_data and not last_net_stats: prompt += "  - (Waiting for initial data)\n"
            else: prompt += "  - (No bandwidth calculated this interval)\n"

            # Add Ping Data
            prompt += f"- Ping to {args.target}: Latency={latency_str}, Loss={loss_str}\n"
            prompt += "\nAnalysis:"

            # --- Call Ollama ---
            ai_response = call_ollama(prompt, args.model)

            # --- Display Analysis ---
            print("\nAI Analysis:")
            if ai_response.startswith("Error:"): print_color(ai_response, COLOR_YELLOW)
            elif ai_response.startswith("ALERT:"): print_color(ai_response, COLOR_RED)
            else:
                if not ai_response.startswith("Status:"): ai_response = "Status: " + ai_response
                print_color(ai_response, COLOR_GREEN)

            # --- Wait ---
            elapsed_time = time.time() - current_time
            sleep_time = max(0, args.interval - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt: print("\nStopping monitor...")
    except Exception as e: print_color(f"\nCritical error in main loop: {e}", COLOR_RED); import traceback; traceback.print_exc()
    finally:
        if nvml_initialized: # Shutdown NVML only if it was successfully initialized
            try: print("Shutting down NVML..."); pynvml.nvmlShutdown()
            except pynvml.NVMLError as error: print_color(f"NVML shutdown error: {error}", COLOR_YELLOW)
            except Exception as e: print_color(f"Unexpected NVML shutdown error: {e}", COLOR_YELLOW)
        print("Exiting.")


if __name__ == "__main__":
    print("Checking dependencies...")
    try: import psutil; import requests; print("- psutil and requests found.")
    except ImportError as e: print_color(f"Error: Missing library: {e.name}. pip install {e.name}", COLOR_RED); sys.exit(1)

    if pynvml_imported: print("- pynvml found (for NVIDIA GPU monitoring).")
    else: print_color("- pynvml not found. GPU monitoring disabled. (Optional: pip install pynvml)", COLOR_YELLOW)

    if platform.system() == "Windows": print("Running on Windows.")
    elif platform.system() in ["Linux", "Darwin"]: print(f"Running on {platform.system()}.")

    main()