"""
Deep learning baselines (BRITS, SAITS) via PyPOTS for spatial extrapolation.

Both methods are real PyTorch models trained with self-supervised learning on
observed sensor data. They demonstrate the fundamental limitation of DL
imputation methods in the sensor deployment (spatial extrapolation) scenario:
without observations at target sensor positions, even trained models cannot
produce meaningful estimates.
"""

from .brits import run_brits, run_saits

DL_METHODS = {
    "brits": run_brits,
    "saits": run_saits,
}
