"""
Storage utility module for NetWorld backend.
Provides a writable temporary directory for uploaded traffic files and artifacts.
Appropriate for serverless execution environments (e.g. AWS Lambda / Vercel where /var/task is read-only)
and local development environments.
"""

import os
import tempfile


def get_upload_dir() -> str:
    """
    Returns a writable temporary directory path for uploaded files.
    Does NOT create the directory at import time.
    """
    if "NETWORLD_UPLOAD_DIR" in os.environ:
        return os.environ["NETWORLD_UPLOAD_DIR"]
    elif os.name != "nt" and os.path.exists("/tmp"):
        return "/tmp/networld_uploads"
    else:
        return os.path.join(tempfile.gettempdir(), "networld_uploads")


def ensure_upload_dir() -> str:
    """
    Lazily creates and returns the writable upload directory.
    Must be called only when write operations actually occur.
    """
    upload_dir = get_upload_dir()
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


# For backward compatibility with modules referencing UPLOAD_DIR as a string
UPLOAD_DIR = get_upload_dir()
