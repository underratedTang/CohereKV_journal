"""CPU-only checks of one-Delta routing/conflict experimental protocol."""
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from threshold_protocol import shared_lambda_at_point


def test_euclidean_sweep_changes_both_routing_and_conflict_coefficient():
    shared = 0.4
    for route_lambda in (0.3, 0.4, 0.5):
        conflict_lambda = shared_lambda_at_point('euclidean', route_lambda, shared)
        assert conflict_lambda == route_lambda


def test_cosine_routing_uses_tau_and_keeps_one_euclidean_delta():
    for cosine_tau in (0.85, 0.9, 0.95):
        assert shared_lambda_at_point('cosine', cosine_tau, 0.4) == 0.4


@pytest.mark.parametrize('metric,param,lam', [('euclidean', -0.1, 0.4),
    ('cosine', 1.5, 0.4), ('cosine', 0.9, 0.0), ('other', 0.5, 0.4)])
def test_invalid_thresholds_fail(metric,param,lam):
    with pytest.raises(ValueError):
        shared_lambda_at_point(metric,param,lam)


def test_shells_expose_one_lambda_and_cosine_tau_only():
    root = HERE.parent
    for path in (HERE / 'run_motivation_attention.sh', root / 'diagnosis_experiments' / 'run_diagnosis.sh'):
        shell = path.read_text()
        assert 'LAMBDA="0.40"' in shell
        assert '--lambda_value "$LAMBDA"' in shell
        assert 'JOURNAL_LAMBDA=' not in shell
        assert 'CONFERENCE_LAMBDA=' not in shell
        assert 'FIG1_CONFLICT_LAMBDA' not in shell
        assert 'FIG2_CONFLICT_LAMBDA' not in shell
        assert 'EUCLIDEAN_ROUTE_LAMBDA' not in shell


def test_csv_and_plot_use_shared_lambda_field():
    src = (HERE / 'motivation_attention_experiment.py').read_text()
    plot = (HERE / 'plot_motivation_attention.py').read_text()
    assert '"lambda_value": float(lambda_value)' in src
    assert 'shared_lambda_at_point(metric, route_param, args.lambda_value)' in src
    assert '"lambda_value"' in plot
    assert '"conflict_lambda"' not in src
    assert '"conflict_lambda"' not in plot
