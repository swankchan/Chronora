1. docker run -p 6379:6379 -d redis:7
-p 1st port: port number of you localhost; second port: port number of Docker container;
-d detached mode (for asynchronous job queue)
redis:7 : Remote Dictationary Server version 7

Start API:
2. uvicorn sdxl_direct:app --host 0.0.0.0 --port 8000

Start worker (can use multiple to cater for concurrent users, 10 for 10 users):
3. rq worker default

CLI API Call:
 curl -X POST http://localhost:8000/api/v1/generate \
  -F "prompt=A modern glass museum, golden hour, 8k" \
  -F "resolution=1024x1024" \
  -F "steps=30" \
  -F "guidance_scale=7.5" \
  -F "batch_size=10" \
  -F "output_format=png"


Frontend for user:
4. cd frontend
5. npm install
6. npm run dev

