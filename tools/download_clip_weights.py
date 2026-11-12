"""Download the official immutable CLIP blob in resumable verified byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
URL = f'https://openaipublic.blob.core.windows.net/clip/models/{EXPECTED}/ViT-L-14-336px.pt'
SIZE = 934088680
CHUNK = 8 * 1024 * 1024
FOLDER = ROOT / 'tmp/clip_weight_chunks'
TARGET = ROOT / 'models/clip/ViT-L-14-336px.verified.pt'


def download(index):
    first, last = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
    path = FOLDER / f'{index:04d}.bin'
    length = last - first + 1
    if path.exists() and path.stat().st_size == length:
        return index
    for retry in range(6):
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={first}-{last}'})
            with urllib.request.urlopen(request, timeout=35) as source:
                if source.status != 206 or source.headers.get('Content-Range') != f'bytes {first}-{last}/{SIZE}':
                    raise RuntimeError('Server did not return the requested immutable range')
                with path.open('wb') as output:
                    while part := source.read(1024 * 1024):
                        output.write(part)
            if path.stat().st_size != length:
                raise RuntimeError('Truncated chunk')
            return index
        except Exception:
            if retry == 5:
                raise
            time.sleep(min(2 ** retry, 10))


if __name__ == '__main__':
    FOLDER.mkdir(parents=True, exist_ok=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    n = (SIZE + CHUNK - 1) // CHUNK
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download, i) for i in range(n)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == n:
                print(f'CLIP download {completed}/{n} chunks; elapsed {time.perf_counter()-started:.0f}s', flush=True)
    digest = hashlib.sha256()
    with TARGET.with_suffix('.partial').open('wb') as output:
        for i in range(n):
            with (FOLDER / f'{i:04d}.bin').open('rb') as source:
                while part := source.read(1024 * 1024):
                    output.write(part)
                    digest.update(part)
    if digest.hexdigest() != EXPECTED:
        raise RuntimeError('Official CLIP SHA256 mismatch')
    TARGET.with_suffix('.partial').replace(TARGET)
    print('Verified official CLIP:', TARGET, digest.hexdigest(), flush=True)
