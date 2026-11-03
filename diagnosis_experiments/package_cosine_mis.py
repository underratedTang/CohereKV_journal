#!/usr/bin/env python3
"""Archive only current experiment outputs; never include previous results or self."""
from pathlib import Path
import argparse,zipfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);a=p.parse_args()
    root=Path(a.input_dir).resolve()
    if not root.is_dir():raise NotADirectoryError(root)
    dest=root/'cosine_mis_median_results.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for file in sorted(root.rglob('*')):
            if file.is_file() and file!=dest and not file.name.startswith('.'):
                out.write(file,file.relative_to(root).as_posix())
    print('[UPLOAD]',dest,'size_mb=',round(dest.stat().st_size/1048576,3),flush=True)
if __name__=='__main__':main()
