# Install and validate

Place this directory at:

```
/home/tlxr/CohereKV/mymethod_cuda_src
```

Install:

```bash
cd /home/tlxr/CohereKV/mymethod_cuda_src
bash install_mymethod_cuda.sh
```

If building on a node where the target GPU is not visible, specify the architecture:

```bash
TORCH_CUDA_ARCH_LIST="8.0" bash install_mymethod_cuda.sh   # A100 example
```

Do not install or import `grouplogitkv_calibrated_qk` for mymethod. GroupLogitKV
may keep its own extension for its independent comparison run.

Then run the integrated engineering gate from the sibling task_evaluation folder:

```bash
cd /home/tlxr/CohereKV/task_evaluation
bash run_mymethod_engineering_validation.sh
```
