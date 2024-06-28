#!/bin/bash
python -m pytest -s -v test_transpose.py test_padding.py test_halo.py test_fft.py test_allgather.py   &> validation_${SLURM_PROCID}.log
