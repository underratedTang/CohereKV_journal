#!/bin/bash

cd /root/SimCalKV

CUDA_VISIBLE_DEVICES=0 \
python test_kv_budget_qwen.py