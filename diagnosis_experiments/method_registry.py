"""Only three baselines and the journal generalization."""
METHODS=('full_cache','eviction_only','merge_all','coherekv_journal')

def select_methods(csv):
    selected=tuple(m.strip() for m in csv.split(',') if m.strip())
    if not selected or len(set(selected))!=len(selected) or any(m not in METHODS for m in selected):
        raise ValueError('Select unique methods from '+','.join(METHODS))
    return tuple(m for m in METHODS if m in selected)

PRIORITY_MODES=('degree','count','importance','count_importance')
