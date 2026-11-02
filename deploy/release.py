"""Model pin validation. Deployment remains disabled until prerequisites are verified."""
import re
import sys

SERVICES = ('embedding', 'reranker', 'nli')
REQUIRED = {
    'embedding': {'model_O4.onnx', 'tokenizer.json'},
    'reranker': {'config.json', 'tokenizer.json', 'model.safetensors'},
    'nli': {'config.json', 'tokenizer.json', 'model.safetensors'},
}


def pins(manifest, selected=SERVICES):
    for service in selected:
        entry = manifest[service]
        if not re.fullmatch(r'[0-9a-f]{40}', entry.get('revision') or ''):
            raise ValueError(f'{service}: missing immutable model revision')
        hashes = entry.get('sha256', {})
        if not REQUIRED[service] <= hashes.keys() or any(
            not re.fullmatch(r'[0-9a-f]{64}', value) for value in hashes.values()
        ):
            raise ValueError(f'{service}: incomplete model checksums')
    return manifest


if __name__ == '__main__':
    print('deploy blocked: runner identity, release units, model provisioning, and authenticated functional health not validated', file=sys.stderr)
    sys.exit(1)
