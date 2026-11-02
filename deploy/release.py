"""Fail-closed station2 release switch. Runner needs pre-provisioned unit/env/models."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

SERVICES = ('embedding', 'reranker', 'nli')
REQUIRED = {
    'embedding': {'model_O4.onnx', 'tokenizer.json'},
    'reranker': {'config.json', 'tokenizer.json', 'model.safetensors'},
    'nli': {'config.json', 'tokenizer.json', 'model.safetensors'},
}


def pins(manifest):
    for service in SERVICES:
        entry = manifest[service]
        if not re.fullmatch(r'[0-9a-f]{40}', entry.get('revision') or ''):
            raise ValueError(f'{service}: missing immutable model revision')
        hashes = entry.get('sha256', {})
        if not REQUIRED[service] <= hashes.keys() or any(
            not re.fullmatch(r'[0-9a-f]{64}', value) for value in hashes.values()
        ):
            raise ValueError(f'{service}: incomplete model checksums')
    return manifest


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    sha = os.environ['DEPLOY_SHA']
    before = os.environ['DEPLOY_BEFORE']
    if not re.fullmatch(r'[0-9a-f]{40}', sha) or not re.fullmatch(r'[0-9a-f]{40}', before):
        raise ValueError('valid immutable before/after SHAs required')
    if run('git', 'rev-parse', 'HEAD') != sha:
        raise ValueError('checkout does not match event SHA')
    root = Path(os.environ['DEPLOY_ROOT']).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('pre-provisioned deployment root required')
    manifest = pins(json.loads(Path('models.json').read_text()))
    changed = run('git', 'diff', '--name-only', before, sha).splitlines()
    selected = set(SERVICES) if any(p.startswith(('deploy/', '.github/workflows/')) or p == 'models.json' for p in changed) else {
        service for service in SERVICES if any(p.startswith(service + '/') for p in changed)
    }
    # Validate every prerequisite before switching any unit. No model download without complete pins.
    for service in selected:
        base = root / service
        if not (base / 'current').is_symlink() or not (base / 'models').is_dir():
            raise ValueError(f'{service}: current symlink/model directory not provisioned')
        for filename, expected in manifest[service]['sha256'].items():
            file = base / 'models' / manifest[service]['revision'] / filename
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{service}: missing or mismatched pinned artifact {filename}')
        if not (base / 'health-url').is_file():
            raise ValueError(f'{service}: health URL not provisioned')
    for service in sorted(selected):
        base = root / service
        current = base / 'current'
        old = current.resolve(strict=True)
        releases = base / 'releases'
        releases.mkdir(exist_ok=True)
        target = releases / sha
        if target.exists():
            raise ValueError(f'{service}: release already exists')
        shutil.copytree(service, target)
        (target / 'model-revision').write_text(manifest[service]['revision'] + '\n')
        pending = base / f'.current-{sha}'
        pending.symlink_to(target)
        unit = f'deepdoc-{service}.service'
        try:
            pending.replace(current)
            run('systemctl', 'restart', unit)
            url = (base / 'health-url').read_text().strip()
            if not url.startswith('https://'):
                raise ValueError('health URL must use HTTPS')
            for _ in range(20):
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f'{service}: health check failed')
        except Exception:
            pending.unlink(missing_ok=True)
            rollback = base / f'.rollback-{sha}'
            rollback.symlink_to(old)
            rollback.replace(current)
            run('systemctl', 'restart', unit)
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'deploy blocked: {exc}', file=sys.stderr)
        sys.exit(1)
