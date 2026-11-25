"""Compact PPL tables; numeric CSVs remain the source of truth."""
import math
WIDTH = 42
PREFIX = '[PPL TABLE] '
def setting(v):
    return '-' if v is None or v == '' else f'{float(v):g}'
def base_header():
    return f'{"Method":<{WIDTH}} {"B":>6} {"p":>5} {"EA":>7} {"EE":>7}'
def base_row(r):
    return f'{r["method"]:<{WIDTH}} {setting(r["actual_budget"]):>6} {setting(r["actual_p"]):>5} {setting(r["actual_lambda_ea"]):>7} {setting(r["actual_lambda_ee"]):>7}'
def number(v):
    return f'{float(v):12.6f}' if v is not None and v != '' else f'{"-":>12}'
def emit_sequence_header():
    h=base_header()+f' {"Seq PPL":>12} {"Tokens":>8} {"Seconds":>10}'
    print(PREFIX+h,flush=True)
    print(PREFIX+'-'*len(h),flush=True)
def emit_sequence_result(r):
    if not r['reused_baseline']:
        print(PREFIX+base_row(r)+' '+number(r['ppl'])+f' {int(r["scored_tokens"]):8d} {float(r["seconds"]):10.2f}',flush=True)
def emit_final_summary(rows,indices,scored_tokens):
    h=f'{"Rank":>4} '+base_header()+f' {"Overall PPL":>12} {"Mean PPL":>12} {"Std PPL":>12}'
    print('\n'+'='*len(h))
    print(f'FINAL SUMMARY | {len(indices)} sequences | indices={",".join(map(str,indices))} | {scored_tokens} scored tokens/config')
    print('Overall PPL = exp(total NLL / total scored tokens); Mean PPL = arithmetic mean; Std PPL = sample SD.')
    print('Ranked by Overall PPL (ascending); unused settings and undefined SD are shown as -.')
    print(h);print('-'*len(h))
    for rank,r in enumerate(sorted(rows,key=lambda r:float(r['overall_ppl'])),1):
        print(f'{rank:4d} '+base_row(r)+' '+' '.join(number(r[k]) for k in ('overall_ppl','mean_ppl','std_ppl')))
    print('='*len(h),flush=True)
