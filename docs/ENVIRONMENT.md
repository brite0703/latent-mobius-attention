# Environment records

configuration/environment_records.json separates original CPU replay, original molecular GPU capture and native N20 capture. Requirements files are exact historical declarations, not a newly tested installation recipe. Preserve CUDA build suffixes, CPU/device scope, thread count, precision and deterministic settings declared by each source.

Current entry checks require Python 3.9 or newer and only its standard library. The bounded checks were run with Python 3.13.5; this does not certify the scientific dependency sets. No current PyTorch/NumPy/RDKit/SciPy/Matplotlib import or clean ML installation test is claimed. Molecular preparation additionally uses RDKit; unrecorded versions are not guessed. Original report modes can require CUDA; CPU replay and original GPU cost profiles are different workloads. Native N20's RTX 3090 / PyTorch 2.9.0+cu128 condition is separate from historical cost measurements.
