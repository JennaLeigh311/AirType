"""
Gunicorn configuration for AirType backend
"""

import multiprocessing
import os

# Server socket
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:5000')
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'geventwebsocket.gunicorn.workers.GeventWebSocketWorker'
worker_connections = 1000
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None
umask = 0

# Logging
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'airtype-backend'

# Server hooks
def pre_fork(server, worker):
    pass

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")
