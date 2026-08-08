# mymethod engineering freeze

Validated on dual RTX 4090 (PyTorch 2.9.0+cu128).

Final production backend for paper experiments:

- certified routing: `torch` (BMM/cuBLAS path)
- conflict graph + parallel MIS: `cuda` (`mymethod_cuda`)
- decode count-calibrated QK: independent `mymethod_cuda` kernel where supported

Reason: the custom routing kernel has much lower workspace but scales poorly in latency, while the CUDA Graph-MIS kernel is 13--44x faster than the PyTorch graph implementation on the validation matrix. The hybrid route=torch, graph=cuda plan is the validated deployment configuration.

Do not use route=`auto` in formal system experiments.
