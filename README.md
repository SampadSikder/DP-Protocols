# Differential Privacy Protocols

This repository contains implementations and demonstrations of various differential privacy (DP) mechanisms, protocols, and attacks. It includes both centralized and local differential privacy techniques, as well as privacy-preserving machine learning methods.

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DP-Protocols
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

Run any of the main demonstration scripts:

```bash
# Test basic DP mechanisms
python noise/basic_dp.py

# Compare Laplace vs Gaussian mechanisms
python ratio_test.py

# Demonstrate local DP protocols
python cfo/oue.py
python cfo/olh.py
python cfo/grr.py

# Show privacy attacks
python demonstrate_attack.py
```
## Running Experiments

### 1. Basic DP Mechanisms Comparison

```bash
python noise/basic_dp.py
```

**Output**: Demonstrates Laplace vs Gaussian mechanisms with privacy-utility tradeoff analysis.

### 2. Empirical DP Verification

```bash
python ratio_test.py
```

**Output**: Tests whether mechanisms satisfy claimed ε-DP using statistical ratio tests.

### 3. Local DP Protocol Comparison

```bash
python cfo/oue.py    # OUE demonstration
python cfo/olh.py    # OLH demonstration  
python cfo/grr.py    # GRR demonstration
```

**Output**: Shows frequency estimation accuracy and variance analysis for each LDP protocol.

### 4. Privacy Attack Demonstration

```bash
python demonstrate_attack.py
```

**Output**: 
- Protocol-agnostic attacks on OUE and OLH
- Attack effectiveness analysis
- Statistical detectability tests

### 5. Advanced DP Techniques

```bash
python exponential_mech.py       # Private selection examples
python range_query_svt.py        # Range queries with SVT
python dp_sgd.py                 # Private machine learning
```

### 6. Custom Experiments

Create your own experiments by importing the mechanisms:

```python
# Example: Custom privacy budget allocation
from noise.basic_dp import LaplaceMechanism
from parallel_composition import parallel_composition_top_k

# Split privacy budget
epsilon_total = 2.0
epsilon_per_query = epsilon_total / 3

# Run multiple queries
results = []
for query in queries:
    mech = LaplaceMechanism(sensitivity=1.0, epsilon=epsilon_per_query)
    noisy_result = mech.add_noise(query(dataset))
    results.append(noisy_result)
```

## Understanding the Output

### Ratio Test Results
- **Test Passed**: ✓/✗ indicates if mechanism satisfies claimed ε-DP
- **Max Ratio Found**: Empirical maximum likelihood ratio
- **Violations**: Number of bins exceeding theoretical bound

### LDP Protocol Metrics
- **True vs Estimated Counts**: Accuracy of frequency estimation
- **Variance Analysis**: Theoretical vs empirical variance
- **Privacy Parameters**: p, q probabilities for each protocol

### Attack Analysis
- **Boost Amount**: Increase in target item's estimated count
- **Boost Percentage**: Relative increase from attack
- **Detectability**: Statistical tests for attack detection

## 🔧 Configuration

### Privacy Parameters
- **epsilon**: Privacy budget (smaller = more private)
- **delta**: Probability of privacy failure (for approximate DP)
- **sensitivity**: Maximum change in function output

### Protocol Parameters
- **d**: Domain size for LDP protocols
- **g**: Hash range for OLH (auto-optimized)
- **k**: Number of categories for GRR

### Attack Parameters
- **target_item**: Item to boost in frequency estimation
- **m_fake_users**: Number of fake users to inject
- **protocol_type**: Target LDP protocol ('OUE', 'OLH', 'GRR')

## Performance Notes

- **Small epsilon**: All protocols add substantial noise
- **Attack detection**: Statistical tests may have false positives
- **Memory usage**: OUE creates d-dimensional vectors per user




