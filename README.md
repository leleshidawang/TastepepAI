
# TastePepAI: An AI-Driven Platform for Taste Peptide De Novo Design

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/yourusername/TastePepAI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.12167-red)](https://arxiv.org/abs/2502.12167)
[![Documentation](https://img.shields.io/badge/Documentation-Available-brightgreen)](http://www.wang-subgroup.com/TastePepAI.html)

## Overview

TastePepAI is a comprehensive artificial intelligence platform designed for customized taste peptide design and safety assessment. This innovative framework addresses the significant challenges in traditional taste peptide discovery by providing an end-to-end automated solution for generating novel taste peptides with desired organoleptic properties while ensuring safety profiles.

The platform integrates advanced machine learning algorithms, including a novel **Loss-supervised Adaptive Variational Autoencoder (LA-VAE)** for controlled peptide generation and **SpepToxPred** for toxicity prediction, enabling researchers to design taste peptides with specific flavor profiles (sweet, salty, umami, sour, bitter) while avoiding unwanted taste characteristics.

## Key Features

- **AI-Driven Peptide Generation**: Utilizes LA-VAE with contrastive learning for precise control of taste properties
- **Comprehensive Safety Assessment**: Integrated toxicity prediction using SpepToxPred with >82% accuracy
- **Taste Avoidance Mechanism**: Selective flavor exclusion capabilities for customized peptide design
- **Automated Workflow**: Complete pipeline from design to physicochemical analysis
- **Multi-taste Capability**: Simultaneous prediction and design of peptides with multiple taste modalities
- **Experimental Validation**: Successfully designed and validated 73 novel taste peptides

## Repository Structure

```
TastePepAI/
├── 01_TastePepAI.py              # Main automated pipeline controller
├── 02_LA-VAE.py                  # Taste peptide design and clustering module
├── 03_TasToxPred.py              # Short peptide toxicity prediction
├── 04_compute_physicochemical.py # Physicochemical property analysis
├── Input_taste_peptides.fasta    # Input file for taste peptide design
├── TasToxPred/                   # Toxicity prediction resources
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages for machine learning and bioinformatics analysis

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/TastePepAI.git
cd TastePepAI

# Install necessary dependencies (numpy, pandas, scikit-learn, tensorflow, etc.)
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn biopython

# Run the complete automated pipeline
python 01_TastePepAI.py
```

## Usage

### Complete Automated Pipeline

For comprehensive taste peptide design including generation, clustering, toxicity prediction, and physicochemical analysis:

```bash
python 01_TastePepAI.py
```

This script automatically executes the following workflow:
1. **Taste Peptide Design & Clustering** - Generates novel peptides with desired taste properties
2. **Toxicity Prediction & Filtering** - Evaluates and filters peptides based on safety profiles  
3. **Physicochemical Property Calculation** - Computes molecular descriptors and properties

### Individual Module Usage

#### Taste Peptide Design and Clustering
```bash
python 02_LA-VAE.py
```
- Input: `Input_taste_peptides.fasta`
- Function: Generates novel taste peptides using LA-VAE architecture
- Output: Clustered peptide sequences with desired taste properties

#### Toxicity Prediction
```bash
python 03_TasToxPred.py
```
- Input: `TasToxPred/` directory contents
- Function: Predicts toxicity of peptide sequences using ensemble learning
- Output: Safety assessment scores and classifications

#### Physicochemical Analysis
```bash
python 04_compute_physicochemical.py
```
- Function: Calculates comprehensive molecular descriptors
- Output: Detailed physicochemical property profiles

## Input Files

- **Input_taste_peptides.fasta**: FASTA format file containing reference taste peptides for model training
- **TasToxPred/**: Directory containing toxicity prediction model files and reference data

## Methodology

### LA-VAE Architecture
- **Loss-supervised Learning**: Dynamic optimization through dual-phase training
- **Contrastive Learning**: Taste avoidance mechanism for selective flavor exclusion
- **Adaptive Generation**: Elastic extension cycles for optimal solution exploration

### SpepToxPred System
- **Feature Engineering**: Integration of 20 sequence encoding descriptors
- **Ensemble Learning**: Weighted voting across 9 machine learning algorithms
- **Specialized Focus**: Optimized for short peptides (≤25 amino acids)

## Scientific Impact

TastePepAI has successfully:
- Designed **73 novel taste peptides** with validated sweet, salty, and umami properties
- Achieved **>90% cell viability** and **<1.5% hemolysis** in safety assessments
- Demonstrated **12.79% improvement** over existing toxicity prediction methods
- Expanded the known sequence space of multi-taste peptides significantly

## Citation
If you use TastePepAI in your research, please see and cite:
https://arxiv.org/abs/2502.12167

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grants No. 32171271, 32271329, and 22473041), the Natural Science Foundation of Hunan Province, and the Scientific Research Program of FuRong Laboratory. We acknowledge the computational resources provided by the Bioinformatics Center of Hunan Normal University and the Key Laboratory of Tea Science of Ministry of Education.

---

**Keywords**: Taste peptides, Artificial intelligence, Peptide design, Food science, Machine learning, Toxicity prediction, Variational autoencoder
