# Fixed Neighbor Search Extension for PyTorch

This repository contains a PyTorch C++ extension for performing efficient fixed-radius neighbor searches in point cloud data. It leverages CUDA for acceleration, making it suitable for high-performance computing tasks in 3D data processing and analysis.

## Features

- **High Performance**: Utilizes CUDA to accelerate the neighbor search process, significantly reducing computation time for large datasets.
- **Easy Integration**: Seamlessly integrates with PyTorch, allowing for direct use within PyTorch pipelines and models.
- **Flexible**: Supports various configurations and parameters to customize the search process according to specific requirements.

## Prerequisites

Before installing the extension, ensure you have the following:

- PyTorch (version >= 1.7.0)
- CUDA Toolkit (version compatible with your PyTorch installation)
- C++ compiler with C++14 support
- Python (version >= 3.6)

## Installation

To install the Fixed Neighbor Search extension, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jselvam11/fixedneighborsearch.git
   cd fixedneighborsearch
2. Build and install the extension
   ```bash
   python setup.py install

## Usage
After installation, you can use the extension in your PyTorch projects as follows:

```python
import torch
import fixed_neighbor_search as fns

# Example usage
points = torch.randn(100, 3, device='cuda')
queries = torch.randn(50, 3, device='cuda')
radius = 0.5

neighbors = fns.fixed_radius_search(points, queries, radius)
