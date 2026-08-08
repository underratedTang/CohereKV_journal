#!/usr/bin/env python3
from pathlib import Path
import re,sys,json

ROOT=Path(__file__).resolve().parent
FILES=[
 ROOT/'models/common/mymethod_common.py',
 ROOT/'models/llama/merge_mymethod_llama.py', ROOT/'models/qwen/merge_mymethod_qwen.py',
 ROOT/'models/llama/mymethod_runtime_patch_llama.py', ROOT/'models/qwen/mymethod_runtime_patch_qwen.py',
]
viol=[]
for p in FILES:
    text=p.read_text()
    # Any executable import/call dependency on GroupLogitKV is forbidden.
    for n,line in enumerate(text.splitlines(),1):
        low=line.lower()
        if ('import ' in low or 'from ' in low) and 'grouplogit' in low:
            viol.append(f'{p.relative_to(ROOT)}:{n}:{line.strip()}')
        if 'grouplogitkv_calibrated_qk' in low:
            viol.append(f'{p.relative_to(ROOT)}:{n}:{line.strip()}')
if viol:
    print(json.dumps({'decoupled':False,'violations':viol},indent=2)); sys.exit(1)
print(json.dumps({'decoupled':True,'checked':[str(p.relative_to(ROOT)) for p in FILES]},indent=2))
