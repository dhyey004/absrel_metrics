# AbsRel Metrics

A lightweight, GPU-accelerated PyTorch micro-library for computing the Absolute Relative Error (AbsRel) in depth estimation models.

# Performance & Vectorization

Computing metrics pixel-by-pixel using standard Python loops is notoriously slow, often taking 50-200ms per batch. This library utilizes **pure PyTorch tensor operations and vectorization** to push calculations directly to the GPU's CUDA cores. 

By executing the math simultaneously in parallel across millions of pixels (e.g., handling spatial masks via `pred[valid_mask]`), execution time drops to **~0.5 to 2.0 milliseconds**. This eliminates CPU bottlenecks and saves hours of computation time over a full training run.

# Installation

We highly recommend installing this package inside a virtual environment to avoid dependency conflicts.

**1. Clone the repository**
```bash
git clone [https://github.com/dhyey004/absrel_metrics.git](https://github.com/dhyey004/absrel_metrics.git)
cd absrel_metrics
