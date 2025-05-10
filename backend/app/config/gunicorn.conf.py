"""
Gunicorn configuration file for YouTube Summarizer API.

This configuration is optimized for resource-constrained environments:
- CPU: 0.1 cores (10% of a single core)
- RAM: 512MB total memory allocation
"""

import multiprocessing
import os

# Server socket
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
backlog = 10  # Maximum number of pending connections

# Worker processes
workers = 2  # Fixed number of workers, not dynamic
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 20  # Maximum number of simultaneous clients per worker
max_requests = 100  # Restart workers after this many requests
max_requests_jitter = 10  # Add randomness to max_requests to prevent all workers from restarting at once

# Process naming
proc_name = "youtube_summarizer"
pythonpath = "."

# Logging
errorlog = "-"  # stderr
accesslog = "-"  # stdout
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Server mechanics
daemon = False
raw_env = []
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Timeouts
timeout = 120  # Worker silent for more than this many seconds is killed
graceful_timeout = 30  # How long to wait for workers to finish their current request before killing them
keepalive = 2  # How long to wait for connections from clients

# Server hooks
def on_starting(server):
    """
    Called just before the master process is initialized.
    """
    pass

def on_reload(server):
    """
    Called to recycle workers during a reload via SIGHUP.
    """
    pass

def when_ready(server):
    """
    Called just after the server is started.
    """
    pass

def pre_fork(server, worker):
    """
    Called just before a worker is forked.
    """
    pass

def post_fork(server, worker):
    """
    Called just after a worker has been forked.
    """
    # Set worker memory limit
    import resource
    # 200MB per worker
    resource.setrlimit(resource.RLIMIT_AS, (200 * 1024 * 1024, 200 * 1024 * 1024))

def pre_exec(server):
    """
    Called just before a new master process is forked.
    """
    pass

def pre_request(worker, req):
    """
    Called just before a worker processes the request.
    """
    worker.log.debug("%s %s" % (req.method, req.path))

def post_request(worker, req, environ, resp):
    """
    Called after a worker processes the request.
    """
    pass

def worker_int(worker):
    """
    Called when a worker receives SIGINT or SIGQUIT.
    """
    worker.log.info("worker received INT or QUIT signal")

def worker_abort(worker):
    """
    Called when a worker receives SIGABRT signal.
    """
    worker.log.info("worker received ABORT signal")

def worker_exit(server, worker):
    """
    Called just after a worker has been exited, in the worker process.
    """
    pass

def nworkers_changed(server, new_value, old_value):
    """
    Called just after num_workers has been changed.
    """
    pass

def on_exit(server):
    """
    Called just before exiting Gunicorn.
    """
    pass
