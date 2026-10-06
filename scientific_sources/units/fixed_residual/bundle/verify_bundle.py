"""Check the supplied residual evidence inventory; no model execution."""
from pathlib import Path, PurePosixPath
import hashlib,json
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'bundle_manifest.json').read_text())
seen=set()
for item in manifest['files']:
    relative=PurePosixPath(item['path'])
    assert not relative.is_absolute() and '..' not in relative.parts and ':' not in str(relative)
    path=(HERE/relative.as_posix()).resolve()
    assert path.is_relative_to(HERE) and item['path'] not in seen
    seen.add(item['path'])
    assert path.stat().st_size==item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
print(json.dumps(dict(passed=True,files=len(seen),bytes=sum(x['bytes'] for x in manifest['files']),
    scope='Inventory, size and SHA-256 integrity only; no new fitting, scoring, native inference or independent metric calculation.')))
