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

#### 2. Behavioral Data (`behavior_data_with_headers_cleaned.csv`)
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

### Processed Data Files (Generated During Analysis)

#### 4. Connectivity Features (`features/conn_matrix_all.npy`)
- **Shape**: (500, 4186) - All subjects × unique connections
- **Content**: Complete functional connectivity matrix
- **Derivation**: Upper triangular correlations between all ROI pairs

#### 5. Behavioral Targets (`target_matrix.npy`)  
- **Shape**: (500, 17) - All subjects × behavioral measures
- **Content**: Preprocessed and cleaned behavioral scores
- **Processing**: Missing value imputation, normalization

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

### Installation and Setup

```bash
# Clone the repository  
git clone https://github.com/yourusername/cogniconnect.git
cd final_cogniconnect_extracted/Final\ Cogniconnect/

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Alternative: Create conda environment
conda create -n cogniconnect python=3.8
conda activate cogniconnect
conda install numpy pandas scikit-learn matplotlib seaborn jupyter

# Launch Jupyter to run notebooks
jupyter notebook
```

### Reproducing the Analysis

1. **Start with Data Exploration**:
   ```bash
   # Open the initial exploration notebook
   jupyter notebook exp1.ipynb
   ```

2. **Run Feature Selection Analysis**:
   ```bash  
   # Analyze connectivity patterns and feature optimization
   jupyter notebook fc_focused.ipynb
   ```

3. **Execute Main Pipeline**:
   ```bash
   # Run the complete analysis pipeline
   jupyter notebook features/final\ final.ipynb
   ```

**Note**: The main fMRI data file (`hcp_data.npy`) must be obtained separately from the Human Connectome Project and placed in the root directory.

### Data Setup

1. **Obtain HCP Data Authorization**
   - Register at [Human Connectome Project](https://www.humanconnectome.org/)
   - Complete data use agreement
   - Download resting-state fMRI data for 500 subjects

2. **Prepare Data Structure**
   ```
   final_cogniconnect_extracted/
   └── Final Cogniconnect/
       ├── hcp_data.npy                          # Main fMRI data (YOU MUST PROVIDE)
       ├── behavior_data_with_headers_cleaned.csv
       ├── roi_info_cleaned.csv
       └── [rest of project structure as shown above]
   ```

3. **Required Data Files**
   - **`hcp_data.npy`**: Original fMRI timeseries (500×92×1200) - **NOT INCLUDED**
   - **`behavior_data_with_headers_cleaned.csv`**: Included - behavioral measures
   - **`roi_info_cleaned.csv`**: Included - brain region information

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

### Running the Analysis Pipeline

**Primary Analysis Notebook**: `features/final final.ipynb`
This is the main notebook containing the complete analysis pipeline.

```python
# Main workflow (conceptual overview of notebook structure)

# 1. Data Loading and Preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Load preprocessed connectivity features and behavioral data
conn_features = np.load('features/conn_matrix_all.npy')  # (500, 4186)
behavioral_data = pd.read_csv('cleaned_behavior_data.csv')
target_matrix = np.load('target_matrix.npy')

# 2. Target-Specific Feature Selection
# For each of the 17 behavioral targets
for target_name in behavioral_targets:
    # Load optimized features for this target
    selected_features = np.load(f'features/final_selected_features/{target_name}_selected.npy')
    
    # Run nested cross-validation
    model = Ridge(alpha=215.44)  # Optimal alpha from hyperparameter tuning
    cv_scores = cross_val_score(model, selected_features, target_scores, cv=5)
    
    print(f"{target_name}: R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Exploring Feature Selection Results

**Feature Selection Exploration**: `fc_focused.ipynb`
```python
# Analyze feature selection optimization results
fc_performance = pd.read_csv('features/topk_fc_per_target/fc_selection_performance.csv')
pca_summary = pd.read_csv('features/topk_fc_per_target/pca_summary_fc.csv')

# Plot optimization curves for different feature counts (10, 20, 30, 40, 50)
import matplotlib.pyplot as plt

for target in targets:
    target_data = fc_performance[fc_performance['target'] == target]
    plt.plot(target_data['n_features'], target_data['cv_score'], 
             label=target, marker='o')

plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation R²')
plt.legend()
plt.title('Feature Selection Optimization')
plt.show()
```

### Initial Exploration and Experimentation

**Exploratory Analysis**: `exp1.ipynb`
```python
# Initial data exploration and method testing
# - Data quality assessment
# - Initial feature extraction experiments  
# - Baseline model comparisons
# - Visualization of brain connectivity patterns

# Load and explore behavioral data
behavior_df = pd.read_csv('behavior_data_with_headers_cleaned.csv')
roi_info = pd.read_csv('roi_info_cleaned.csv')

# Basic statistics and correlations
print("Behavioral data shape:", behavior_df.shape)
print("Missing values:", behavior_df.isnull().sum())

# Correlation matrix of behavioral measures
correlation_matrix = behavior_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Behavioral Measures Correlation Matrix')
plt.show()
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

## 🔧 Key Notebooks and Files

### 📊 Core Analysis Files

#### 1. **`features/final final.ipynb`** - Main Analysis Pipeline
- **Purpose**: Complete brain-behavior prediction pipeline
- **Key Components**:
  - Nested cross-validation implementation
  - Target-specific feature selection
  - Model comparison (Ridge, Lasso, ElasticNet, Random Forest, SVR)
  - Statistical validation with permutation testing
  - Final performance evaluation
- **Outputs**: 
  - `nested_cv_results_all_targets.csv`
  - Individual feature files in `final_selected_features/`

#### 2. **`fc_focused.ipynb`** - Functional Connectivity Analysis
- **Purpose**: Deep dive into connectivity patterns and optimization
- **Key Components**:
  - Feature selection strategy comparison
  - PCA dimensionality reduction analysis
  - Connectivity matrix visualization
  - Network-specific analysis
- **Outputs**:
  - `fc_selection_performance.csv`
  - `pca_summary_fc.csv`
  - Top-k feature files in `topk_fc_per_target/`

#### 3. **`exp1.ipynb`** - Initial Exploration
- **Purpose**: Data exploration and preliminary analysis
- **Key Components**:
  - Data quality assessment
  - Behavioral correlations analysis
  - Initial connectivity extraction
  - Baseline model testing
- **Outputs**: Exploratory plots and initial findings

### 📈 Results and Performance Files

#### **`model_performance_results.csv`**
Contains final performance metrics for all 17 behavioral targets:
```csv
target,best_model,best_r2,r2_std,best_params,feature_selection_method
Flanker_AgeAdj,Ridge,0.1713,0.0379,alpha=215.44,f_regression
Flanker_Unadj,Ridge,0.1437,0.0584,alpha=215.44,f_regression
ProcSpeed_Unadj,Ridge,0.1295,0.0435,alpha=215.44,f_regression
...
```

#### **`feature_selection_comparison.csv`**
Comparison of feature selection strategies across targets:
```csv
target,f_regression_score,mutual_info_score,lasso_score,rf_importance_score,best_method
CardSort_AgeAdj,0.1162,0.0892,0.1089,0.1043,f_regression
PicVocab_Unadj,0.0876,0.0798,0.0924,0.0812,lasso
SCPT_SEN,0.0234,0.0287,0.0198,0.0341,rf_importance
...
```

#### **`nested_cv_results_all_targets.csv`**
Detailed cross-validation results with fold-by-fold performance:
```csv
target,fold,train_r2,test_r2,model_type,n_features,alpha
Flanker_AgeAdj,1,0.1856,0.1642,Ridge,30,215.44
Flanker_AgeAdj,2,0.1789,0.1721,Ridge,30,215.44
...
```

### 🧬 Processed Feature Files

#### **`features/conn_matrix_all.npy`**
- **Shape**: (500, 4186) 
- **Content**: Complete functional connectivity matrix for all subjects
- **Description**: All unique pairwise correlations between 92 brain regions
- **Calculation**: Upper triangular of correlation matrices (excluding diagonal)

#### **`features/final_selected_features/`**
Target-specific optimized features:
- **Naming Convention**: `{TARGET_NAME}_selected.npy`
- **Typical Shape**: (500, 20-40) depending on target
- **Selection Method**: Varies by target (F-regression, Lasso, or Random Forest)
- **Optimization**: Based on cross-validation performance

#### **`features/topk_fc_per_target/`**
Feature count optimization results:
- **File Pattern**: `{TARGET_NAME}_top{K}.npy` where K ∈ {10, 20, 30, 40, 50}
- **Purpose**: Systematic evaluation of optimal feature count
- **Analysis**: Performance vs. feature count trade-off curves

## 📁 Project Structure

```
cogniconnect_extracted/
└── Cogniconnect/
    ├── behavior_data_with_headers_cleaned.csv    # Original behavioral data
    ├── cleaned_behavior_data.csv                 # Preprocessed behavioral measures
    ├── roi_info_cleaned.csv                      # Brain region information (92 ROIs)
    ├── target_matrix.npy                         # Processed behavioral targets matrix
    │
    ├── exp1.ipynb                                # Initial exploration and experimentation
    ├── fc_focused.ipynb                          # Functional connectivity analysis notebook
    │
    ├── features/
    │   ├── conn_matrix_all.npy                   # Full connectivity matrices (500×4186)
    │   ├── final final.ipynb                     # Main analysis pipeline notebook
    │   ├── nested_cv_results_all_targets.csv     # Cross-validation results summary
    │   │
    │   ├── final_selected_features/              # Optimal features per target
    │   │   ├── CardSort_AgeAdj_selected.npy      # Executive function features
    │   │   ├── CardSort_Unadj_selected.npy
    │   │   ├── Flanker_AgeAdj_selected.npy       # Inhibitory control features
    │   │   ├── Flanker_Unadj_selected.npy
    │   │   ├── Language_Task_Acc_selected.npy    # Language processing features
    │   │   ├── ListSort_AgeAdj_selected.npy      # Working memory features
    │   │   ├── ListSort_Unadj_selected.npy
    │   │   ├── MMSE_Score_selected.npy           # General cognition features
    │   │   ├── PicVocab_AgeAdj_selected.npy      # Vocabulary features
    │   │   ├── PicVocab_Unadj_selected.npy
    │   │   ├── ProcSpeed_AgeAdj_selected.npy     # Processing speed features
    │   │   ├── ProcSpeed_Unadj_selected.npy
    │   │   ├── ReadEng_AgeAdj_selected.npy       # Reading ability features
    │   │   ├── ReadEng_Unadj_selected.npy
    │   │   ├── SCPT_SEN_selected.npy             # Attention sensitivity features
    │   │   ├── SCPT_SPEC_selected.npy            # Attention specificity features
    │   │   └── WM_Task_Acc_selected.npy          # Working memory accuracy features
    │   │
    │   └── topk_fc_per_target/                   # Feature selection optimization
    │       ├── CardSort_AgeAdj_top10.npy         # Top 10 features for each target
    │       ├── CardSort_AgeAdj_top20.npy         # Top 20 features for each target
    │       ├── CardSort_AgeAdj_top30.npy         # Top 30 features for each target
    │       ├── CardSort_AgeAdj_top40.npy         # Top 40 features for each target
    │       ├── CardSort_AgeAdj_top50.npy         # Top 50 features for each target
    │       ├── [... similar files for all 17 targets ...]
    │       ├── fc_selection_performance.csv       # Feature count optimization results
    │       └── pca_summary_fc.csv                # PCA dimensionality reduction summary
    │
    ├── feature_selection_comparison.csv          # Comparison of selection strategies
    ├── model_performance_results.csv             # Final model performance metrics
    └── optimization_results.png                  # Visualization of optimization process

# Note: hcp_data.npy (original fMRI data) not included due to confidentiality
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

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

**Important**: While the code is open source, access to the Human Connectome Project data requires separate authorization and agreement to their data use terms.

## 📞 Contact

**Shweta Shardul**  
Email: svs28@njit.edu  
Institution: New Jersey Institute of Technology  
Advisor: Professor Mengjia Xu  

**Project Links**:
- GitHub Repository: https://github.com/shwetashardul/CogniConnect/

## 🙏 Acknowledgments

- **Human Connectome Project** for providing high-quality neuroimaging data
- **Professor Mengjia Xu** for guidance and supervision
- **NJIT High-Performance Computing Facility** for computational resources
- **Open-source community** for the excellent machine learning tools that made this work possible

---

*CogniConnect represents a significant step forward in understanding the neural basis of individual cognitive differences through advanced machine learning approaches. We hope this work contributes to the broader goals of computational neuroscience and cognitive enhancement research.*
