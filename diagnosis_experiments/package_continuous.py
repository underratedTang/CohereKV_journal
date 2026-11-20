#!/usr/bin/env python3
"""Package all current continuous PPL numeric/log/figure outputs, once."""
import argparse
from pathlib import Path
import zipfile


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--input_dir',required=True)
    a=cli.parse_args()
    root=Path(a.input_dir).resolve()
    if not root.is_dir():raise NotADirectoryError(root)
    output=root/'cohere_p_continuous_results.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(root.rglob('*')):
            if file.is_file() and file!=output and not file.name.startswith('.'):
                archive.write(file,file.relative_to(root).as_posix())
    print('[UPLOAD]',output,'size_MB',round(output.stat().st_size/1048576,3),flush=True)


if __name__=='__main__':main()
