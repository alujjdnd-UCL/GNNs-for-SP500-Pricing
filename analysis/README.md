# Analysis and Visualization Scripts

This directory contains scripts for post-processing experimental results and generating visualizations presented in the paper.

## Prerequisites

Before running analysis scripts, ensure:
1. Experiments have been completed (`experiments/metrics.csv` exists)
2. Graph adjacency matrices have been saved (`data/graphs/` directory populated)
3. Required packages are installed (included in main `requirements.txt`)

## Available Scripts

### Model Performance Analysis

#### `model_comparison.py`
Generates comparative visualizations of model performance across different metrics.

**Usage**:
```bash
python model_comparison.py
```

**Outputs**:
- Line plots comparing RMSE/MAE/MRE across models
- Bar charts for portfolio performance
- Phase-wise performance breakdown

**Requires**: `experiments/metrics.csv`

---

#### `metrics_heatmap.py`
Creates heatmap visualizations of model performance metrics.

**Usage**:
```bash
python metrics_heatmap.py
```

**Outputs**:
- Heatmaps showing model × phase performance
- Correlation matrices between different metrics

---

### Graph Structure Analysis

#### `graph_centrality.py`
Analyzes centrality measures of constructed graphs.

**Usage**:
```bash
python graph_centrality.py
```

**Outputs**:
- Degree centrality distributions
- Betweenness centrality rankings
- Eigenvector centrality analysis

**Requires**: Graph adjacency matrices in `data/graphs/`

---

#### `graph_centrality_heatmap.py`
Generates heatmap visualizations of node centralities across different graph construction methods.

**Usage**:
```bash
python graph_centrality_heatmap.py
```

**Outputs**:
- Centrality heatmaps by stock
- Comparative centrality across edge formation strategies

---

#### `graph_centrality_indiv.py`
Analyzes centrality measures for individual stocks across phases.

**Usage**:
```bash
python graph_centrality_indiv.py
```

**Outputs**:
- Stock-specific centrality trends over time
- Top-k central stocks identification

---

#### `centrality_top.py`
Identifies and visualizes top-k most central stocks in each graph.

**Usage**:
```bash
python centrality_top.py
```

**Outputs**:
- Rankings of most central stocks
- Sector distribution of central nodes

---

#### `graph_degrees.py`
Analyzes degree distributions of constructed graphs.

**Usage**:
```bash
python graph_degrees.py
```

**Outputs**:
- Degree distribution histograms
- Summary statistics (mean degree, density)

---

### Graph Construction Comparison

#### `graph_construction_violin.py`
Creates violin plots comparing different graph construction strategies.

**Usage**:
```bash
python graph_construction_violin.py
```

**Outputs**:
- Violin plots of edge weights
- Distribution comparisons across strategies

---

#### `graph_cons_error_megaplot.py`
Generates comprehensive comparison plots of graph construction methods vs. model errors.

**Usage**:
```bash
python graph_cons_error_megaplot.py
```

**Outputs**:
- Multi-panel plots showing:
  - Error rates by graph construction method
  - Model performance across strategies
  - Phase-wise trends

---

### Market Conditions Analysis

#### `market_conditions.py`
Analyzes market conditions during different phases.

**Usage**:
```bash
python market_conditions.py
```

**Outputs**:
- Volatility measures by phase
- Return distributions
- Market regime classification

**Requires**: `data/SP500/raw/phases/*.csv`

---

#### `market_conditions_plot.py`
Generates visualizations of market conditions over time.

**Usage**:
```bash
python market_conditions_plot.py
```

**Outputs**:
- Time series plots of market indicators
- Phase boundaries marked on plots
- Volatility regime highlighting

---

## Common Usage Patterns

### Generate All Visualizations
```bash
cd analysis
for script in *.py; do
    echo "Running $script..."
    python "$script"
done
```

### Generate Specific Figure from Paper
```bash
# Figure 1: Model comparison
python model_comparison.py

# Figure 2: Graph structure analysis
python graph_centrality_heatmap.py

# Figure 3: Market conditions
python market_conditions_plot.py
```

---

## Output Locations

By default, scripts save outputs to:
- **Figures**: Current directory or `figures/` subdirectory
- **Data tables**: CSV files in current directory

Modify output paths by editing the `OUTPUT_DIR` variable in each script.

---

## Customization

### Modify Plotting Style
Edit the matplotlib style settings at the top of each script:
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')  # Change to your preferred style
```

### Change Figure Resolution
```python
plt.savefig('output.png', dpi=300)  # Adjust dpi value
```

### Filter Specific Models/Phases
Most scripts support filtering via command-line arguments or by editing the script's configuration section.

---

## Dependencies

These scripts use:
- **matplotlib**: Plotting
- **seaborn**: Statistical visualizations
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **networkx**: Graph analysis (for centrality scripts)
- **scipy**: Statistical computations

All dependencies are included in the main `requirements.txt`.

---

## Troubleshooting

### Issue: "FileNotFoundError: metrics.csv"
**Solution**: Run experiments first (`cd experiments && python run_all.py`)

### Issue: Empty plots or missing data
**Solution**: Ensure all experiment phases completed successfully. Check `experiments/metrics.csv` for completeness.

### Issue: "No module named 'networkx'"
**Solution**: Install missing dependency: `pip install networkx`

### Issue: Large memory usage
**Solution**: Some scripts load all graph data into memory. Process phases individually or increase available RAM.

---

## Adding New Analysis Scripts

To add custom analysis:
1. Create a new `.py` file in this directory
2. Import required libraries and load data:
   ```python
   import pandas as pd
   metrics = pd.read_csv('../experiments/metrics.csv')
   ```
3. Perform analysis and generate visualizations
4. Save outputs and document usage in this README

---

## Reproducing Paper Figures

| Figure # | Script | Description |
|----------|--------|-------------|
| 1 | `model_comparison.py` | Model performance comparison |
| 2 | `graph_centrality_heatmap.py` | Graph centrality analysis |
| 3 | `graph_construction_violin.py` | Edge formation comparison |
| 4 | `market_conditions_plot.py` | Market regime visualization |
| 5 | `graph_cons_error_megaplot.py` | Comprehensive error analysis |

---

## Contact

For questions about specific visualization scripts or to request additional analyses, please contact the authors or open an issue on the repository.
