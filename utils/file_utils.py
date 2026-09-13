import hashlib
from pathlib import Path

def file_fingerprint(file_path):
    if file_path is None:
        return None
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def get_file_extension(file_path):
    if not file_path:
        return ""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path):
    return get_file_extension(file_path) in {".pdf", ".doc", ".docx"}