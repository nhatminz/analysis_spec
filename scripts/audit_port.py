#!/usr/bin/env python3
"""Verify local port integrity and, optionally, the untouched read-only reference."""
from pathlib import Path
import argparse
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference')
    args=parser.parse_args();manifest=json.loads((ROOT/'PORT_SOURCES.json').read_text());failed=[]
    for row in manifest['ports']:
        path=ROOT/row['destination']
        if not path.is_file() or sha(path)!=row['destination_sha256']:failed.append(row['destination'])
    if args.reference:
        before=json.loads((ROOT/'validation/reference_before.json').read_text());reference=Path(args.reference)
        after={str(p.relative_to(reference)):sha(p) for p in reference.rglob('*') if p.is_file() and
               '__pycache__' not in p.parts and '.git' not in p.parts}
        if before!=after:failed.extend('reference:'+p for p in sorted(before.keys()|after.keys()) if before.get(p)!=after.get(p))
    if failed:raise SystemExit('Port/reference integrity mismatch: '+', '.join(failed))
    print(f'{len(manifest["ports"])} local source mappings verified; '+('reference unchanged' if args.reference else 'no reference directory required'))

if __name__=='__main__':main()
