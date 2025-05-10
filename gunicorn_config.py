# gunicorn_config.py

# Bind to 0.0.0.0 so it's accessible from outside the container
# The port should match what your Dockerfile EXPOSEs and what Nginx proxies to
bind = "0.0.0.0:5002"

# Number of worker processes
# A common recommendation is (2 * number_of_cores) + 1
# Given your docker-compose limits cpus to 0.5, start with a small number.
# You can adjust this based on performance monitoring.
workers = 2  # Start with 2 workers

# Worker class
# 'gthread' is good for I/O-bound applications (like Flask apps making API calls)
# For gthread, you can also specify threads per worker
threads = 6  # Number of threads per worker
worker_class = "gthread"

# Timeout for workers (in seconds)
# Adjust if you have long-running requests
timeout = 1200

# Logging
# Send access and error logs to stdout/stderr so Docker can capture them
accesslog = "-"
errorlog = "-"
loglevel = "info" # Can be 'debug', 'info', 'warning', 'error', 'critical'

# Optional: Preload app for faster worker forking (can have memory implications)
# preload_app = True

# Optional: Max requests a worker will process before restarting
# Helps prevent memory leaks in long-running applications
# max_requests = 1000
# max_requests_jitter = 50 # Add some randomness to restarts

# The WSGI application object
# Assumes your Flask app instance is named 'app' in 'quality_compute_simulator.py'
# This is often not needed here if specified in the CMD, but good for clarity
# raw_env = ["FLASK_APP=quality_compute_simulator:app"] # Not typically needed with direct app module call