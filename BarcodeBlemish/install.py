
import os.path as osp
import shutil

src_dir = osp.dirname(__file__)
dst_dir = osp.join(src_dir, '../projects/DNS')

mapping = {
    osp.join(src_dir, 'README.md'):  osp.join(dst_dir, 'README.md'),
    osp.join(src_dir, 'notebooks/example.ipynb'): osp.join(dst_dir, 'notebooks/example.ipynb'),
}

for src, dst in mapping.items():
    print(f'[SRC] {src} -> [DST] {dst}')
    shutil.copy(src, dst)
