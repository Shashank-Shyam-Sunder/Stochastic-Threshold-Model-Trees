# Stochastic Threshold Model Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

## Overview

Stochastic Threshold Model Trees is a novel tree-based ensemble method designed to provide reasonable extrapolation predictions for physicochemical and other data that are expected to have a certain degree of monotonicity. This method addresses the challenging problem of making reliable predictions outside the training data range, which is crucial in many scientific and engineering applications.

## Key Features

- **Extrapolation-focused**: Specifically designed to handle predictions beyond the training data range
- **Stochastic approach**: Uses probabilistic threshold selection for improved robustness
- **Ensemble method**: Combines multiple tree models for better generalization
- **Monotonicity preservation**: Maintains expected monotonic relationships in the data
- **Flexible integration**: Compatible with various regression models and splitting criteria

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Examples and Visualizations](#examples-and-visualizations)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees.git
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees.git
cd Stochastic-Threshold-Model-Trees
```

2. Install in development mode:
```bash
pip install -e .
```

## Requirements

### Core Dependencies
- **Python** >= 3.6
- **NumPy** >= 1.17 - For numerical computations
- **joblib** == 0.13 - For parallel processing
- **scikit-learn** == 0.21 - For machine learning utilities

### Optional Dependencies (for notebooks and examples)
- **pandas** >= 0.25 - For data manipulation
- **matplotlib** >= 3.1 - For plotting and visualization
- **seaborn** >= 0.9 - For enhanced statistical visualizations

### Installation of All Dependencies

```bash
# Core dependencies only
pip install numpy>=1.17 joblib==0.13 scikit-learn==0.21

# All dependencies including optional ones
pip install numpy>=1.17 joblib==0.13 scikit-learn==0.21 pandas>=0.25 matplotlib>=3.1 seaborn>=0.9
```

## Quick Start

```python
from StochasticThresholdModelTrees.regressor.stmt import StochasticThresholdModelTrees
from StochasticThresholdModelTrees.threshold_selector import NormalGaussianDistribution
from StochasticThresholdModelTrees.criterion import MSE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('./data/logSdataset1290.csv', index_col=0)
X = data[data.columns[1:]]
y = data[data.columns[0]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = StochasticThresholdModelTrees(
    n_estimators=100,
    criterion=MSE(),
    regressor=LinearRegression(),
    threshold_selector=NormalGaussianDistribution(5),
    random_state=42
)

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Detailed Usage

### Model Parameters

The `StochasticThresholdModelTrees` class accepts the following parameters:

- **n_estimators** (int, default=100): The number of regression trees to create in the ensemble
- **criterion** (object): Criteria for setting divisional boundaries (e.g., MSE())
- **regressor** (object): Regression model applied to each terminal node (e.g., LinearRegression())
- **threshold_selector** (object): Parameters for determining candidate division boundaries
- **min_samples_leaf** (float, default=1.0): Minimum number of samples required to make up a leaf node
- **max_features** (str or int, default='auto'): Number of features to consider for optimal splitting
- **f_select** (bool, default=True): Whether to choose features to consider when splitting
- **ensemble_pred** (str, default='median'): Aggregation method during ensemble ('mean' or 'median')
- **scaling** (bool, default=False): Whether to perform standardization as preprocessing to each terminal node
- **random_state** (int, default=None): Random state for reproducibility

### Threshold Selectors

Currently supported threshold selectors:

- **NormalGaussianDistribution(sigma)**: Uses Gaussian distribution with specified sigma value

### Splitting Criteria

Available splitting criteria:

- **MSE()**: Mean Squared Error criterion

## Examples and Visualizations

The method demonstrates superior performance in extrapolation scenarios, as shown in the following examples:

### Discontinuous Function Prediction
![discontinuous_Proposed_5sigma](https://user-images.githubusercontent.com/49966285/86465964-ad039700-bd6d-11ea-80b0-8035fc726228.png)

The proposed method successfully captures the underlying trend even in discontinuous regions.

### Multi-dimensional Sphere Function
![Sphere_Proposed_MLR_noise_scaling](https://user-images.githubusercontent.com/49966285/86466391-7d08c380-bd6e-11ea-879c-8e9b3f9ba493.png)

Performance comparison on sphere function with noise, demonstrating robust extrapolation capabilities.

### 1D Function Comparison
![1dim_comparison](https://user-images.githubusercontent.com/49966285/86992420-69c97e00-c1dc-11ea-8e2f-8b3d08ce27d4.png)

Direct comparison with other methods on 1-dimensional functions, highlighting the advantages in extrapolation regions.

## API Reference

### Core Classes

#### `StochasticThresholdModelTrees`

Main estimator class implementing the stochastic threshold model trees algorithm.

**Methods:**
- `fit(X, y)`: Fit the model to training data
- `predict(X)`: Make predictions on new data
- `score(X, y)`: Return the coefficient of determination R² of the prediction

#### `NormalGaussianDistribution`

Threshold selector using normal Gaussian distribution.

**Parameters:**
- `sigma` (float): Standard deviation parameter for threshold selection

#### `MSE`

Mean Squared Error criterion for node splitting.

## Contributing

We welcome contributions to improve the Stochastic Threshold Model Trees project! Here's how you can contribute:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/Stochastic-Threshold-Model-Trees.git
cd Stochastic-Threshold-Model-Trees

# Install in development mode with all dependencies
pip install -e .
pip install pandas>=0.25 matplotlib>=3.1 seaborn>=0.9

# Run tests (when available)
python -m pytest
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all public methods
- Include tests for new features
- Update documentation as needed
- Ensure backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

The MIT License is a permissive license that allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

With the following conditions:
- 📋 Include license and copyright notice
- ❌ No liability or warranty

## Citation

If you use Stochastic Threshold Model Trees in your research, please cite our paper:

```bibtex
@article{stochastic_threshold_trees,
    title={Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation},
    author={Authors},
    journal={arXiv preprint arXiv:2009.09171},
    year={2020},
    url={https://arxiv.org/abs/2009.09171}
}
```

**Paper Reference:**
[Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation](https://arxiv.org/abs/2009.09171)

## Contact

For questions, suggestions, or collaboration opportunities:

- **Issues**: [GitHub Issues](https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees/issues)
- **Discussions**: [GitHub Discussions](https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees/discussions)

## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special acknowledgment to the research community for valuable feedback
- Built with ❤️ for the machine learning community

---

**Note**: This project is actively maintained. For the latest updates and features, please check our [GitHub repository](https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees).