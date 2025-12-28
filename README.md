# Modeling EEG Responses using Amplitude-Binned Envelope Features with mTRF Framework

A MATLAB-based research project for analyzing EEG data using Linear regression model - the multivariate Temporal Response Function (mTRF) framework - with novel amplitude-binned (AB) envelope features.

## 📖 Overview

This project investigates cortical tracking of complex sound envelopes by extracting audio features (envelope, onset, amplitude-binned envelope) and correlating them with EEG recordings. The core innovation is the **Amplitude-Binned (AB) Envelope** method, which bins audio amplitude into discrete ranges to better capture neural responses to sound intensity changes.

For detailed information, see the [Munich Brain Day 2025 Poster](Munich_Brain_Day_2025_Poster.pdf). or [Munich Brain Day 2025 Abstract Book](MBD25_abstractbook_final2.pdf).

Full report is here: [Amplitude-Binned Envelope Modeling Report](Report - Amplitude-Binned Envelope Modeling for EEG Response Prediction in Naturalistic Soundscapes (Internal Research Project - IRP).pdf)




![poster for Munich_Brain_Day_2025](Munich_Brain_Day_2025_Poster.png)

---

## 🎯 Key Features

- **Audio Feature Extraction**: Standard envelope, onset detection, and amplitude-binned envelope generation
- **EEG Preprocessing**: Automated pipeline for XDF-to-SET conversion, ICA, and artifact removal
- **mTRF Analysis**: Forward and backward encoding models to predict neural responses from audio features
- **Grid Search Optimization**: Systematic parameter tuning for bin edges and number of bins
- **Statistical Analysis**: Correlation analysis, permutation testing, and visualization tools
- **Topographic Brain Maps**: Visualization of activation patterns across scalp electrodes

---

## 📁 Project Structure

```
├── src/
│   ├── main/                    # Main analysis pipelines
│   ├── feature_extraction/      # Audio feature generation
│   ├── analysis/                # Statistical analysis & mTRF modeling
│   ├── preprocessing/           # EEG preprocessing functions
│   ├── visualization/           # Plotting and topoplot functions
│   └── utils/                   # Helper functions (data loading, etc.)
├── data/                        # Input data files
├── results/                     # Generated figures and outputs
├── docs/                        # Documentation and reference materials
└── archive/                     # Archived/deprecated code
```

---

## 🚀 Getting Started

### Prerequisites

- **MATLAB** R2019b or later
- **EEGLAB** (2024.2 recommended)
- **mTRF Toolbox** (Multivariate Temporal Response Function Toolbox)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Elnurimamaliyev/Project-EEG-ABenvelope.git
   cd Project-EEG-ABenvelope
   ```

2. **Download dependencies:**
   - EEGLAB: https://sccn.ucsd.edu/eeglab/
   - mTRF Toolbox: https://github.com/mickcrosse/mTRF-Toolbox

3. **Configure paths:**
   Edit `src/preprocessing/OT/OT_setup.m` to set:
   - EEGLAB installation path
   - mTRF Toolbox path
   - Data directories (DATAPATH, raw_data_path)

### Running the Analysis

1. **Preprocess EEG data:**
   ```matlab
   run('src/main/PreprocessingScript.m')
   ```
   This converts raw XDF files to EEGLAB format, performs ICA, and saves preprocessed data.

2. **Run main analysis:**
   ```matlab
   run('src/main/MainScript.m')
   ```
   Extracts audio features, trains mTRF models, and generates correlation statistics.

3. **Visualize results:**
   ```matlab
   run('src/analysis/Plots_and_Statistics.m')
   ```
   Creates plots comparing feature performance across subjects and conditions.

4. **Parameter optimization (optional):**
   ```matlab
   run('src/main/MainScript_SearchGrid.m')
   ```
   Performs grid search over bin edges and number of bins.

---

## 🧪 Core Components

### Audio Features

- **Envelope** (`mTRFenvelope`): Broadband audio amplitude envelope
- **Onset** (`OnsetGenerator.m`): Detects acoustic onset events
- **AB Envelope** (`ABenvelopeGenerator_V2.m`): Amplitude-binned envelope with configurable bins

### Analysis Pipeline

1. **Data Loading** (`LoadEEG.m`): Loads preprocessed EEG and audio
2. **Feature Extraction**: Generates multiple audio features
3. **mTRF Modeling** (`mTRFrun.m`): Trains forward/backward models
4. **Statistical Testing**: Correlation analysis, t-tests, permutation tests
5. **Visualization**: Topoplots, correlation boxplots, weight matrices

### Key Scripts

| Script | Purpose |
|--------|---------|
| `MainScript.m` | Primary analysis pipeline |
| `MainScript_new_features.m` | Extended feature testing |
| `MainScript_SearchGrid.m` | Parameter optimization |
| `PreprocessingScript.m` | EEG preprocessing |
| `Plots_and_Statistics.m` | Statistical visualization |
| `mTRFrun.m` | mTRF model training and evaluation |

---

## 📊 Expected Outputs

- **Correlation values** (Pearson r) for each feature and subject
- **Model weights** showing temporal response patterns
- **Topoplots** of activation across scalp electrodes
- **Statistical comparisons** between narrow/wide audio conditions
- **Grid search results** for optimal binning parameters

---

## 🔧 Configuration

Edit `src/preprocessing/OT/OT_setup.m` to configure:

```matlab
% Toolbox paths
addpath(genpath('C:\Path\To\EEGLAB\'));
addpath(genpath('C:\Path\To\mTRF-Toolbox\'));

% Data paths
DATAPATH = 'C:\Path\To\EEG\Data\';
raw_data_path = 'P:\';

% Subject and task lists
sbj = {'P002', 'P019', 'P009', 'P006', 'P020', 'P012'};
task = {'narrow', 'wide'};
```

---

## 👥 Authors

- **Thorge Haupt** (Original implementation)
- **Elnur Imamaliyev** (Modifications and extensions)

**Affiliation**: University of Oldenburg, Neuroscience Department

**Dataset**: `\\daten.uni-oldenburg.de\psychprojects$\Neuro\Thorge Haupt\data\Elnur`

---

## 📚 References

- **Crosse et al. (2016)**: "The Multivariate Temporal Response Function (mTRF) Toolbox"
- **Drennan & Lalor (2019)**: "Cortical Tracking of Complex Sound Envelopes: Modeling the Changes in Response with Intensity"

See `docs/reference_study/` for reference materials.

---

## 📝 Notes

- **Channel Selection**: Analysis uses 5 central electrodes (C3, FC2, FC1, Cz, C4)
- **Time Window**: mTRF models typically use -100 to 400 ms window
- **Regularization**: Lambda = 0.05 (adjustable in `mTRFrun.m`)
- **Normalization**: Features are range-normalized before modeling

---

## 🗂️ Archive

The `archive/` folder contains deprecated code, old versions, and backup files not needed for current analyses.

---

## 📄 License

This project is part of academic research at the University of Oldenburg.

---

## 🔗 Links

- **GitHub Repository**: https://github.com/Elnurimamaliyev/Project-EEG-ABenvelope
- **Munich Brain Day 2025 Poster**: [View Poster](Munich_Brain_Day_2025_Poster.pdf)
