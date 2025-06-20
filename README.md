# CogniConnect: Brain-Behavior Prediction from Functional Connectivity

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-red.svg)](https://numpy.org)

**Predicting Behavioral Measures from Brain Functional Connectivity: A Machine Learning Approach**

CogniConnect is a comprehensive machine learning framework that predicts individual cognitive performance across multiple domains from resting-state brain functional connectivity patterns. Using data from the Human Connectome Project, this system achieves up to 17% variance explained in cognitive measures, with executive function showing the strongest brain-behavior relationships.

## 🧠 Project Overview

### What is CogniConnect?

CogniConnect bridges the gap between brain organization and cognitive function by leveraging advanced machine learning techniques to predict individual differences in cognitive abilities from functional brain connectivity patterns. The project addresses fundamental questions in cognitive neuroscience:

- **How much** of cognitive performance can be predicted from brain connectivity?
- **Which** cognitive domains show the strongest neural relationships?
- **What** machine learning approaches work best for brain-behavior prediction?
- **How** do different feature engineering strategies impact accuracy?

### Key Achievements

- **17% variance explained** in executive function measures (Flanker task)
- **Comprehensive analysis** across 17 behavioral measures spanning 5 cognitive domains
- **Methodological innovation** with target-specific feature selection
- **High-performance computing** optimization for large-scale neuroimaging analysis
- **Rigorous validation** with nested cross-validation and 1,000 permutation tests

## 📊 Dataset Information

### Data Sources (Human Connectome Project)

**⚠️ Important Note**: The main neuroimaging data file (`hcp_data.npy`) is not included in this repository due to data sensitivity and confidentiality agreements with the Human Connectome Project. To reproduce this analysis, you must obtain proper authorization and download the data directly from the HCP.

#### 1. Brain fMRI Data (`hcp_data.npy`) - **NOT INCLUDED**
- **Subjects**: 500 healthy adults
- **Brain Regions**: 92 ROIs per subject
- **Temporal Resolution**: 1200 time points (14.4 minutes, TR=0.72s)
- **Preprocessing**: Z-normalized BOLD signals
- **Size**: ~2.6GB (500 × 92 × 1200 array)

#### 2. Behavioral Data (`behavior_data_with_headers_cleaned-2.csv`)
- **Features**: 35 behavioral/demographic measures
- **Cognitive Domains**:
  - **Executive Function**: CardSort, Flanker tasks
  - **Working Memory**: WM_Task_Acc, ListSort
  - **Attention**: SCPT_SEN, SCPT_SPEC  
  - **Processing Speed**: ProcSpeed
  - **Language**: ReadEng, PicVocab
  - **General Cognition**: MMSE

#### 3. ROI Information (`roi_info_cleaned.csv`)
- **Brain Region Details**: 92 anatomical regions
- **Network Affiliation**: VisCent, DefaultB, SalVentAttnA, etc.
- **Spatial Information**: Hemisphere (LH/RH), RGB coordinates
- **Visualization Support**: For network analysis and plotting

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# High-performance computing
joblib>=1.1.0
multiprocessing
```

### Installation

```bash
# Clone the repository
git clone https://github.com/shwetashardul/cogniconnect.git
cd cogniconnect

# Install dependencies
pip install -r requirements.txt

# For HPC environments with conda
conda env create -f environment.yml
conda activate cogniconnect
```

### Data Setup

1. **Obtain HCP Data Authorization**
   - Register at [Human Connectome Project](https://www.humanconnectome.org/)
   - Complete data use agreement
   - Download resting-state fMRI data for 500 subjects

2. **Prepare Data Structure**
   ```
   cogniconnect/
   ├── data/
   │   ├── hcp_data.npy                    # Main fMRI data (YOU MUST PROVIDE)
   │   ├── behavior_data_with_headers_cleaned-2.csv
   │   └── roi_info_cleaned.csv
   ├── src/
   ├── results/
   └── README.md
   ```

## 🔬 Methodology Deep Dive

### 1. High-Performance Computing Architecture

**Resource Optimization Strategy**:
- **CPU Utilization**: 120 cores (93.75% of available resources)
- **GPU Acceleration**: 4 GPUs for parallel processing
- **Memory Management**: Batch processing (50 subjects per batch)
- **Parallelization Levels**:
  - 80% resources → fMRI feature extraction
  - 50% resources → concurrent model training  
  - 20% resources → behavioral target processing

**Why This Approach?**
The computational demands of analyzing 500 subjects × 92 brain regions × 1200 time points (≈55 million data points) necessitated sophisticated resource management. Our multi-level parallelization strategy reduces processing time from days to hours while maintaining numerical stability.

### 2. Feature Engineering Pipeline

#### A. Connectivity Feature Extraction

**Method 1: High-Variance Selection**
```python
# Conceptual workflow
for subject in subjects:
    correlation_matrix = compute_connectivity(subject_timeseries)
    high_variance_features = select_top_k_features(correlation_matrix, k=5)
```

**Method 2: Comprehensive Target-Specific Selection**
```python
# Extract all possible connections
total_connections = (n_regions * (n_regions - 1)) / 2  # 4,186 unique connections

# For each behavioral target
for target in behavioral_targets:
    selected_features = feature_selector.fit_transform(
        connectivity_features, target_scores
    )
```

**Why Target-Specific Selection?**
Different cognitive functions engage distinct brain networks. Executive function might rely on prefrontal-parietal connections, while language processing depends on temporal-frontal pathways. Target-specific selection captures these domain-specific neural signatures.

#### B. Dimensionality Reduction Strategy

**Principal Component Analysis Implementation**:
- **Variance Retention**: 95% of original variance
- **Typical Reduction**: 25-30% dimensionality decrease
- **Adaptive Thresholding**: Target-specific component selection

**Mathematical Justification**:
```
PCA Transformation: X_reduced = X × W_k
where W_k contains top k eigenvectors explaining ≥95% variance
```

This reduces overfitting risk while preserving essential connectivity patterns that drive cognitive performance.

### 3. Feature Selection Methodology Comparison

#### Strategies Evaluated:

**1. F-Regression (Winner for 13/17 targets)**
- **Principle**: Univariate linear relationships
- **Advantage**: Captures direct linear brain-behavior associations
- **Mathematical Foundation**: F-statistic testing for linear correlation significance

**2. Lasso Regularization (Optimal for vocabulary measures)**
- **Principle**: L1 penalty induces sparsity
- **Advantage**: Identifies minimal critical connections
- **Result**: PicVocab required only 6-8 connections (vs. 30-50 for other domains)

**3. Random Forest Importance (Best for attention measures)**
- **Principle**: Tree-based feature importance
- **Advantage**: Captures non-linear interaction effects
- **Insight**: Attention networks show complex, hierarchical organization

**4. Mutual Information**
- **Principle**: Non-linear dependency measurement
- **Application**: Secondary validation of linear findings

### 4. Model Architecture and Validation

#### Nested Cross-Validation Framework

```
Outer Loop (5-fold CV): Performance Estimation
├── Fold 1: Train on 80%, Test on 20%
├── Fold 2: Train on 80%, Test on 20%
├── ...
└── Inner Loop (3-fold CV): Hyperparameter Optimization
    ├── Grid Search over parameter space
    └── Select optimal configuration
```

**Why Nested CV?**
- **Outer loop**: Provides unbiased performance estimates
- **Inner loop**: Optimizes model parameters without data leakage
- **Result**: Generalizable performance metrics

#### Model Comparison Results

| Model | Cognitive Domain Strength | Key Characteristics |
|-------|--------------------------|-------------------|
| **Ridge Regression** | Universal winner (16/17 targets) | L2 regularization, linear relationships |
| ElasticNet | Moderate performance | Combined L1+L2 penalties |
| Lasso | Vocabulary-specific | Sparse feature selection |
| Random Forest | Attention measures | Non-linear interactions |
| SVR | Limited success | Kernel-based non-linearity |

**Ridge Regression Dominance**:
- **Optimal α = 215.44** across most targets
- **Interpretation**: Substantial regularization needed to prevent overfitting
- **Implication**: Brain-behavior relationships are predominantly linear

## 📈 Results and Performance

### Cognitive Domain Hierarchy

| Rank | Cognitive Measure | R² Score | Model | Key Insight |
|------|------------------|----------|-------|-------------|
| 1 | **Flanker_AgeAdj** | **0.171** | Ridge | Executive function shows strongest neural coupling |
| 2 | Flanker_Unadj | 0.144 | Ridge | Age adjustment enhances prediction |
| 3 | ProcSpeed_Unadj | 0.130 | Ridge | Processing speed highly connected to brain networks |
| 4 | ReadEng_Unadj | 0.126 | Ridge | Language abilities show stable neural patterns |
| 5 | ListSort_Unadj | 0.117 | Ridge | Working memory moderately predictable |

### Statistical Validation

- **Permutation Testing**: 1,000 permutations confirm statistical significance
- **Cross-Validation Stability**: Consistent performance across folds
- **Effect Size**: 10-17% variance explained represents medium-to-large effects in neuroscience

### Network-Specific Insights

**Most Predictive Brain Networks**:
1. **Salience/Ventral Attention A** (9 regions): Executive control
2. **Default Mode Network B** (8 regions): Self-referential processing  
3. **Visual Central** (7 regions): Perceptual processing
4. **Dorsal Attention B** (7 regions): Spatial attention

## 🧪 Usage Examples

### Basic Prediction Pipeline

```python
# Load preprocessed data
connectivity_features = load_connectivity_features()
behavioral_targets = load_behavioral_data()

# Initialize CogniConnect pipeline
from cogniconnect import BrainBehaviorPredictor

predictor = BrainBehaviorPredictor(
    feature_selection='target_specific',
    model_type='ridge',
    validation='nested_cv'
)

# Fit model for specific cognitive domain
predictor.fit(connectivity_features, behavioral_targets['Flanker_AgeAdj'])

# Generate predictions
predictions = predictor.predict(new_connectivity_data)
performance = predictor.evaluate()
```

### Advanced Feature Analysis

```python
# Analyze feature importance across cognitive domains
from cogniconnect import FeatureAnalyzer

analyzer = FeatureAnalyzer()
importance_map = analyzer.compute_domain_specific_importance(
    connectivity_features, 
    all_behavioral_targets
)

# Visualize brain networks
analyzer.plot_network_importance(importance_map, roi_info)
```

### High-Performance Computing Configuration

```python
# Configure parallel processing
from cogniconnect import HPCConfig

config = HPCConfig(
    n_cpu_cores=120,
    n_gpu_devices=4,
    batch_size=50,
    memory_limit='32GB'
)

# Run full analysis pipeline
results = run_full_analysis(
    hcp_data_path='data/hcp_data.npy',
    config=config,
    output_dir='results/'
)
```

## 📊 Interpreting Results

### Performance Metrics

**R² (Coefficient of Determination)**:
- **0.17**: Excellent for neuroimaging (Flanker task)
- **0.10-0.13**: Good predictive performance
- **<0.05**: Limited predictability (attention measures)

**Statistical Significance**:
- All reported results significant at p < 0.001
- Permutation testing controls for multiple comparisons
- Cross-validation ensures generalizability

### Cognitive Domain Insights

**Executive Function (Highest Predictability)**:
- **Neural Basis**: Stable prefrontal-parietal network connectivity
- **Implication**: Executive control relies on consistent network architecture
- **Clinical Relevance**: Potential biomarker for executive dysfunction

**Attention (Lowest Predictability)**:
- **Neural Basis**: Dynamic, state-dependent processing
- **Implication**: Resting-state connectivity insufficient for attention prediction
- **Future Direction**: Task-based connectivity analysis needed

**Language Processing (Moderate Predictability)**:
- **Neural Basis**: Left-hemisphere temporal-frontal connections
- **Feature Sparsity**: Only 6-8 connections needed (Lasso selection)
- **Interpretation**: Language has focal neural architecture

## 🔧 Technical Implementation Details

### Memory Management

**Large Data Handling**:
```python
# Efficient memory usage for 2.6GB+ datasets
def process_subjects_in_batches(data, batch_size=50):
    n_subjects = data.shape[0]
    for start_idx in range(0, n_subjects, batch_size):
        end_idx = min(start_idx + batch_size, n_subjects)
        yield data[start_idx:end_idx]
```

**Connectivity Computation Optimization**:
```python
# Parallel correlation matrix computation
from joblib import Parallel, delayed

def compute_connectivity_parallel(timeseries_data, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(np.corrcoef)(subject_data.T) 
        for subject_data in timeseries_data
    )
```

### Hyperparameter Optimization

**Ridge Regression Tuning**:
```python
# Logarithmic alpha search space
alpha_range = np.logspace(-3, 3, 50)  # 0.001 to 1000

# HalvingGridSearchCV for efficiency
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

search = HalvingGridSearchCV(
    Ridge(), 
    {'alpha': alpha_range},
    factor=2,
    cv=3
)
```

## 📁 Project Structure

```
cogniconnect/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing/
│   │   ├── connectivity_extraction.py
│   │   ├── behavioral_processing.py
│   │   └── quality_control.py
│   ├── feature_engineering/
│   │   ├── feature_selection.py
│   │   ├── dimensionality_reduction.py
│   │   └── target_specific_selection.py
│   ├── modeling/
│   │   ├── brain_behavior_predictor.py
│   │   ├── model_comparison.py
│   │   └── validation_framework.py
│   ├── analysis/
│   │   ├── performance_evaluation.py
│   │   ├── feature_importance.py
│   │   └── statistical_testing.py
│   └── visualization/
│       ├── brain_network_plots.py
│       ├── performance_visualization.py
│       └── feature_maps.py
├── data/
│   ├── hcp_data.npy                    # NOT INCLUDED - MUST OBTAIN FROM HCP
│   ├── behavior_data_with_headers_cleaned-2.csv
│   ├── roi_info_cleaned.csv
│   └── processed/                      # Generated during analysis
├── results/
│   ├── model_performance/
│   ├── feature_importance/
│   ├── statistical_validation/
│   └── visualizations/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_results_analysis.ipynb
├── requirements.txt
├── environment.yml
└── README.md
```

## 🎯 Key Findings and Implications

### Scientific Contributions

1. **Methodological Innovation**:
   - Target-specific feature selection outperforms universal approaches
   - Linear models (Ridge) consistently superior to complex non-linear methods
   - Substantial regularization (α ≈ 215) required for optimal generalization

2. **Neuroscientific Insights**:
   - Executive function most tightly coupled to functional connectivity
   - Age-adjusted measures sometimes more predictable than raw scores
   - Vocabulary skills depend on sparse, specific neural connections

3. **Clinical Implications**:
   - Functional connectivity biomarkers feasible for cognitive assessment
   - Domain-specific prediction strategies needed for optimal accuracy
   - Resting-state fMRI sufficient for some but not all cognitive domains

### Theoretical Impact

**Linear Brain-Behavior Relationships**:
The dominance of Ridge regression challenges assumptions about non-linear neural complexity, suggesting that fundamental cognitive-neural relationships may be more parsimonious than commonly assumed.

**Network Specialization vs. Integration**:
Differential predictability across domains supports theories of specialized neural systems while highlighting the importance of network integration for complex cognition.

## 🚀 Future Directions

### Immediate Extensions

1. **Longitudinal Analysis**:
   - Track brain-behavior relationships over time
   - Investigate developmental and aging effects
   - Assess stability of predictive features

2. **Task-Based Connectivity**:
   - Extend beyond resting-state to task-evoked connectivity
   - Compare resting vs. task predictive power
   - Domain-specific task connectivity analysis

3. **Clinical Applications**:
   - Validate in clinical populations
   - Develop diagnostic and prognostic tools
   - Test intervention effects on brain-behavior relationships

### Advanced Methodological Development

1. **Dynamic Connectivity**:
   - Time-varying connectivity patterns
   - State-dependent prediction models
   - Temporal dynamics of brain-behavior coupling

2. **Multi-Modal Integration**:
   - Combine structural and functional connectivity
   - Include diffusion tensor imaging data
   - Integrate genetic and molecular markers

3. **Causal Inference**:
   - Causal discovery algorithms
   - Neuromodulation validation studies
   - Intervention-based causal testing

**Key References**:
- Human Connectome Project: https://www.humanconnectome.org/
- Van Essen, D.C., et al. (2013). The WU-Minn Human Connectome Project. NeuroImage.
- Finn, E.S., et al. (2015). Functional connectome fingerprinting. Nature Neuroscience.

## 🤝 Contributing

We welcome contributions to improve CogniConnect! Areas for contribution:

- **Bug fixes and optimizations**
- **Additional feature selection methods**
- **New visualization capabilities**  
- **Documentation improvements**
- **Testing and validation**

Please see `CONTRIBUTING.md` for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

**Important**: While the code is open source, access to the Human Connectome Project data requires separate authorization and agreement to their data use terms.

## 📞 Contact

**Shweta Shardul**  
Email: svs28@njit.edu  
Institution: New Jersey Institute of Technology  
Advisor: Professor Mengjia Xu  


## 🙏 Acknowledgments

- **Human Connectome Project** for providing high-quality neuroimaging data
- **Professor Mengjia Xu** for guidance and supervision
- **NJIT High-Performance Computing Facility** for computational resources
- **Open-source community** for the excellent machine learning tools that made this work possible

---

*CogniConnect represents a significant step forward in understanding the neural basis of individual cognitive differences through advanced machine learning approaches. We hope this work contributes to the broader goals of computational neuroscience and cognitive enhancement research.*
