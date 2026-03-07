#!/bin/sh
set -e

# Write runtime env.js consumed by the frontend bundle.
# If VITE_API_BASE is set, inject it into /usr/share/nginx/html/env.js

ENV_FILE=/usr/share/nginx/html/env.js
echo "/* Runtime environment injected by docker-entrypoint */" > $ENV_FILE
echo "window.__env__ = {" >> $ENV_FILE
echo "  VITE_API_BASE: \"${VITE_API_BASE:-http://localhost:8001}\"" >> $ENV_FILE
echo "};" >> $ENV_FILE

exec "$@"
