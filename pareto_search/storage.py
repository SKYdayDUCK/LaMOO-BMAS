"""Finite JSON persistence. One owning process per optimizer directory."""
import hashlib
import json
import os
from pathlib import Path


def stable_hash(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_suffix(path.suffix+'.tmp')
    with pending.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,path)


def freeze_json(path,payload):
    if Path(path).exists():
        if stable_hash(read_json(path)) != stable_hash(payload):
            raise ValueError('Frozen configuration differs: '+str(path))
    else:
        write_json(path,payload)
