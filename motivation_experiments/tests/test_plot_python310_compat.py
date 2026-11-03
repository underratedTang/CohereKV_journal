"""Regression for the Python 3.10 f-string/backslash plot SyntaxError."""
import ast
from pathlib import Path


def test_motivation_plot_parses_under_python_310_grammar():
    file = Path(__file__).resolve().parents[1] / 'plot_motivation_attention.py'
    ast.parse(file.read_text(encoding='utf-8'), filename=str(file), feature_version=(3, 10))
