#!/bin/bash
# Master script to run all experiments sequentially on cloud server.
# Step 1a: HPO for CPU methods
# Step 1b: HPO for DL methods
# Step 2: Experiment 1 (main benchmark)
# Step 3: Experiment 2 (ablation)
# Step 4: Experiment 3 (sensitivity)
# Step 5: Experiment 4 (convergence)

set -e

PROJECT=/root/project
cd $PROJECT

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

echo "============================================"
echo "  Starting all experiments"
echo "  Time: $(date)"
echo "============================================"

# Step 1a: HPO for CPU methods
echo ""
echo ">>> Step 1a: Running HPO for CPU methods..."
python run_hpo.py --max-iter 200 --cache-dir results/hpo_cache

echo ""
echo ">>> Step 1a complete: $(date)"

# Step 1b: HPO for DL methods
echo ""
echo ">>> Step 1b: Running HPO for DL methods..."
python run_hpo.py --methods brits saits --max-iter 200 --cache-dir results/hpo_cache

echo ""
echo ">>> Step 1b complete: $(date)"

# Step 2: Experiment 1 (main benchmark, all 11 methods)
echo ""
echo ">>> Step 2: Running Experiment 1 (main benchmark)..."
python run_experiment1.py --deploy-numbers 20 50 100 150 190 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment1.json

echo ""
echo ">>> Step 2 complete: $(date)"

# Step 3: Experiment 2 (ablation)
echo ""
echo ">>> Step 3: Running Experiment 2 (ablation)..."
python run_experiment2.py --deploy-numbers 20 50 100 150 190 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment2.json

echo ""
echo ">>> Step 3 complete: $(date)"

# Step 4: Experiment 3 (sensitivity)
echo ""
echo ">>> Step 4: Running Experiment 3 (parameter sensitivity)..."
python run_experiment3.py --n-deploy 100 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment3.json

echo ""
echo ">>> Step 4 complete: $(date)"

# Step 5: Experiment 4 (convergence)
echo ""
echo ">>> Step 5: Running Experiment 4 (convergence)..."
python run_experiment4.py --deploy-numbers 50 100 150 --gammas 0.5 1.0 5.0 --max-iter 200 \
    --output results/experiment4.json

echo ""
echo ">>> Step 5 complete: $(date)"

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Time: $(date)"
echo "============================================"
echo ""
echo "Results files:"
ls -la $PROJECT/results/*.json 2>/dev/null
