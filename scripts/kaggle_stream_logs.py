import time
import threading
import queue
import kaggle

api = kaggle.KaggleApi()
api.authenticate()
kernel = "anilmittal/name-that-llm-lora-peft-training"

status = api.kernels_status(kernel)
print(f"Status: {status}", flush=True)

msg_queue = queue.Queue()

def stream_worker():
    try:
        for item in api.kernels_logs_stream(kernel):
            msg_queue.put(item)
    except Exception as e:
        msg_queue.put({"error": str(e)})

t = threading.Thread(target=stream_worker, daemon=True)
t.start()

lines = []
start_time = time.time()
last_recv_time = time.time()

while time.time() - start_time < 15:
    try:
        item = msg_queue.get(timeout=2.0)
        last_recv_time = time.time()
        txt = item.get("data", "")
        if txt:
            for line in txt.splitlines():
                if line.strip():
                    lines.append(f"[{item.get('stream_name', '')} {item.get('time', 0):.1f}s] {line}")
    except queue.Empty:
        # If we already got messages and haven't gotten any in 2 seconds, we are at the tail of the stream
        if lines and (time.time() - last_recv_time > 2.0):
            break
        continue

print(f"Total lines captured: {len(lines)}", flush=True)
print("--- Last 40 lines ---", flush=True)
for l in lines[-40:]:
    print(l, flush=True)
