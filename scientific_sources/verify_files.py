"""Verify supplied files without executing scientific code or models."""
from pathlib import Path, PurePosixPath
import hashlib, json, sys

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=set(); errors=[]
    for row in data['files']:
        rel=PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):
            raise ValueError('Unsafe manifest path')
        if str(rel) in expected: raise ValueError('Duplicate manifest path')
        expected.add(str(rel)); p=root.joinpath(*rel.parts)
        if not p.is_file(): errors.append({'path':str(rel),'error':'missing'})
        elif p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            errors.append({'path':str(rel),'error':'content differs'})
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    errors += [{'path':p,'error':'unlisted file'} for p in sorted(actual-expected-{'manifest.json'})]
    print(json.dumps({'passed':not errors,'files_checked':len(expected),'errors':errors,
                     'scope':'File presence, size and SHA-256 only; no inference or training.'},indent=2))
    if errors: sys.exit(1)
if __name__=='__main__': main()
