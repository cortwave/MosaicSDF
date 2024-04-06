from pathlib import Path

import objaverse.xl as oxl

from tqdm import tqdm
from scripts.optimize_msdf import optimize_msdf
import hashlib

CACHE_DIR = './.objaverse'


def get_hash(inp: str) -> str:
    input_bytes = inp.encode('utf-8')
    hash_object = hashlib.sha256()
    hash_object.update(input_bytes)
    hex_digest = hash_object.hexdigest()
    return hex_digest


def process(df):
    downloaded = oxl.download_objects(df, download_dir=CACHE_DIR)
    print('downloaded', len(downloaded.keys()))

    for identifier, local_path in tqdm(downloaded.items()):
        try:
            hash_name = get_hash(identifier)
            save_hash_name = output_path / f'{hash_name}.pt'
            if save_hash_name.exists():
                continue
            optimize_msdf(input_path=local_path, output_path=save_hash_name)
            with open('embeddings.txt', 'a') as f:
                f.write(f'{identifier} {save_hash_name}\n')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    annotations = oxl.get_alignment_annotations(download_dir=CACHE_DIR)
    output_path = Path('msdf_embeddings')
    output_path.mkdir(exist_ok=True)

    sketch_df = annotations[annotations['source'] == 'sketchfab']
    bs = 100
    start_index = 0
    for i in tqdm(range(start_index, len(sketch_df), bs)):
        end = min(i + bs, len(sketch_df))
        process(sketch_df[i:end])


