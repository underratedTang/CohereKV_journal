"""Compact table; CSV stores the complete configuration identity."""
PREFIX='[PPL TABLE] '

def cell(v):return '-' if v=='' or v is None else str(v)

def header():return f'{"ID":<7} {"Method":<20} {"Budget":>6} {"Ref":>12} {"A_EA":>6} {"A_EE":>6} {"L_EA":>6} {"L_EE":>6} {"Priority":>16}'
def base(r):return f'{r["config_id"]:<7} {r["method"]:<20} {cell(r["budget"]):>6} {cell(r["reference_mode"]):>12} {cell(r["alpha_ea"]):>6} {cell(r["alpha_ee"]):>6} {cell(r["lambda_ea"]):>6} {cell(r["lambda_ee"]):>6} {cell(r["priority_mode"]):>16}'
def emit_sequence_header():print(PREFIX+header()+f' {"PPL":>12} {"Tokens":>8} {"Seconds":>10}',flush=True)
def emit_sequence_result(r):print(PREFIX+base(r)+f' {float(r["ppl"]):12.6f} {int(r["scored_tokens"]):8d} {float(r["seconds"]):10.2f}',flush=True)
def emit_final_summary(rows,indices,scored):
    print(f'FINAL SUMMARY | windows={indices} | {scored} scored tokens/config')
    print(header()+f' {"Overall PPL":>12} {"Mean PPL":>12} {"Std PPL":>12}')
    for r in sorted(rows,key=lambda x:float(x['overall_ppl'])):
        print(base(r)+f' {float(r["overall_ppl"]):12.6f} {float(r["mean_ppl"]):12.6f} {cell(r["std_ppl"]):>12}')
