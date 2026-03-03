import os
import redis
from rq import Queue, Connection, SimpleWorker
from dotenv import load_dotenv

# Load .env file (if present) for local development
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Socket timeout for redis client (seconds, float)
REDIS_SOCKET_TIMEOUT = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "5"))
# RQ default timeout for this queue (seconds)
RQ_DEFAULT_TIMEOUT = int(os.environ.get("RQ_DEFAULT_TIMEOUT", "21600"))

redis_conn = redis.from_url(redis_url, socket_timeout=REDIS_SOCKET_TIMEOUT)

if __name__ == '__main__':
    print("Starting RQ worker for Chronora tasks...\nConnect to:", redis_url)
    print(f"[worker] socket_timeout={REDIS_SOCKET_TIMEOUT}; RQ default_timeout={RQ_DEFAULT_TIMEOUT}")
    with Connection(redis_conn):
        q = Queue("default", connection=redis_conn, default_timeout=RQ_DEFAULT_TIMEOUT)
        worker = SimpleWorker(["default"], connection=redis_conn)
        worker.work()
