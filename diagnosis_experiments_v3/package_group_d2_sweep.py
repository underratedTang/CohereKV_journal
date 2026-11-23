#!/usr/bin/env python3
"""Compact unified upload ZIP for parameter sweep, retaining key diagnostics.

Full v17-style raw fusion pair and anchor CSVs remain available in the local
per-setting ZIPs. This sweep ZIP intentionally excludes huge raw per-pair
logs, all nested ZIP files and other nonessential diagnostics.
"""
from __future__ import annotations
import argparse,zipfile
from pathlib import Path

ROOT_NAMES={
  'run.log','sweep_plan.csv','group_d2_sweep_manifest.json',
  'group_d2_sweep_settings.csv','group_d2_sweep_pooled_methods.csv',
  'group_d2_sweep_per_window.csv','group_d2_sweep_ranked.csv',
  'group_d2_sweep_rounds.csv'
}
SETTING_NAMES={
  'group_d2_manifest.json','group_d2_per_sequence.csv',
  'group_d2_all_methods.csv','group_d2_pooled_methods.csv',
  'group_d2_paired_contrasts.csv', 'group_d2_round_metrics.csv'
}
SEQUENCE_NAMES={
  'continuous_config.json','continuous_run_complete.json',
  'ppl_continuous.csv','ppl_continuous_cycles.csv',
  'ppl_continuous_head_rounds.csv',
}


def include(rel):
    parts=rel.parts
    if len(parts)==1:return parts[0] in ROOT_NAMES
    if parts[0]=='sweep_figures' and len(parts)==2:
        return rel.suffix in ('.pdf','.png')
    if parts[0]=='calibration' and len(parts)==2:
        return parts[1] in ('median_reference.json','median_reference_by_head.csv')
    if parts[0]!='settings' or len(parts)<3:return False
    if len(parts)==3:return parts[2] in SETTING_NAMES
    if len(parts)==4 and parts[2]=='aggregate_figures':
        return rel.suffix in ('.pdf','.png')
    if len(parts)>=6 and parts[2]=='sequences' and parts[4]=='ppl_continuous':
        if len(parts)==6:return parts[5] in SEQUENCE_NAMES
        if len(parts)==7 and parts[5]=='figures':return rel.suffix in ('.pdf','.png')
    return False


def pack(root,allow_partial=False):
    root=Path(root).resolve()
    if not root.is_dir():raise NotADirectoryError(root)
    completed=root/'group_d2_sweep_manifest.json'
    if not allow_partial and not completed.is_file():
        raise RuntimeError('Missing completed sweep manifest; refusing to label partial sweep complete')
    out=root/'cohere_p_continuous_results.zip'
    count=0
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zipf:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file==out or file.name.startswith('.'):
                continue
            rel=file.relative_to(root)
            if include(rel):
                zipf.write(file,rel.as_posix());count+=1
    if count==0:raise RuntimeError('No numeric results available to package')
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Corrupt sweep ZIP member '+bad)
    print('[UPLOAD]',out,'files',count,'size_MB',round(out.stat().st_size/1048576,2),
          'complete',completed.exists(),flush=True)
    return out


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input_dir',required=True)
    a.add_argument('--allow_partial',type=int,choices=(0,1),default=0)
    args=a.parse_args()
    pack(args.input_dir,bool(args.allow_partial))


if __name__=='__main__':main()
