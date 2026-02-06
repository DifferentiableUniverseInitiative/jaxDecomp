#!/bin/bash
python -m pytest -s -v test_transpose.py test_halo.py test_fft.py  &> validation_${SLURM_PROCID}.log
