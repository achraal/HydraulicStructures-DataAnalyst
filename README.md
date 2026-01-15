# HydraulicStructures-DataAnalyst

A comprehensive data analysis platform for hydraulic structures, featuring advanced statistical and machine learning analysis with a modern desktop application interface.

## 📋 Project Overview

HydraulicStructures-DataAnalyst is a specialized tool designed to analyze hydraulic engineering data through multivariate statistical methods and machine learning techniques. The project processes data from hydraulic channels and structures, providing detailed insights through:

- **Principal Component Analysis (PCA)** - Dimensionality reduction and variance analysis
- **Correspondence Analysis (AFC)** - Chi-square analysis and factorial plane visualization
- **Clustering Analysis** - K-means clustering with quality metrics
- **Machine Learning** - Random Forest prediction models
- **Anomaly Detection** - Isolation Forest and Local Outlier Factor for cybersecurity anomaly detection
- **Correlation Analysis** - Heatmaps and correlation matrices

## 🎯 Features

### Statistical Analysis
- Descriptive statistics (mean, standard deviation)
- Correlation matrix generation
- Centered and reduced matrix transformations
- Quality of representation metrics
- Contribution analysis for individuals and variables

### Visualization
- Factorial planes for PCA and AFC
- Correlation circles
- Heatmaps with color-coded values
- Cluster distribution charts
- Scatter plots for anomaly detection

### Data Processing
- Excel file support for hydraulic data import
- Frequency matrix analysis
- Chi-square testing and interpretation
- Data normalization and standardization

### Machine Learning
- Random Forest models for predictive analytics
- Prediction on new individuals
- Ensemble-based anomaly detection
- Multi-algorithm anomaly detection (IF + LOF)

## 🖥️ Desktop Application Transition

This project has transitioned from a command-line Python script to a **fully-fledged desktop application** using:

- **CustomTkinter** - Modern GUI framework with dark mode support
- **PyInstaller** - Executable packaging for standalone distribution
- **Matplotlib Integration** - Embedded data visualization in the UI

### Running as Desktop App

The application is built as a Windows desktop executable with PyInstaller. To compile the application:

```bash
pyinstaller --onedir --console --clean --exclude-module PyQt6 --exclude-module PySide6 --hidden-import=data_ouvrageHydraulique --hidden-import=numpy --hidden-import=scipy --collect-all numpy --collect-all scipy --add-data "*.xlsx;." interface.py
```

After compilation, copy required MKL libraries to the distribution folder:

```batch
cd dist\interface
copy "C:\Users\<YourUsername>\anaconda3\Library\bin\mkl*.dll" .
copy "C:\Users\<YourUsername>\anaconda3\Library\bin\libiomp5md.dll" .
copy "C:\Users\<YourUsername>\anaconda3\Library\bin\libmmd.dll" .
copy "C:\Users\<YourUsername>\anaconda3\Library\bin\svml_dispmd.dll" .
```

The resulting executable in `dist\interface\interface.exe` can be run standalone without Python installation.

## 📁 Project Structure

```
HydraulicStructures/
├── interface.py                      # Main GUI application (CustomTkinter)
├── data_ouvrageHydraulique.py        # Core analysis functions
├── codeHydraulique.py                # Hydraulic-specific analysis routines
├── contingence.py                    # Contingency table handling
├── cs.py                             # Utility functions
├── dataset.py                        # Data management
├── exports/                          # Generated analysis outputs
│   ├── composantes_individus.csv     # Individual component scores
│   ├── composantes_variables.csv     # Variable component scores
│   ├── contribution_individus.csv    # Individual contributions
│   ├── contribution_variables.csv    # Variable contributions
│   ├── matrice_centree_reducee.csv   # Centered/reduced matrix
│   ├── matrice_correlation.csv       # Correlation matrix
│   ├── pourcentages_clusters_k*.csv  # Cluster distributions
│   └── qualite_representation.csv    # Quality metrics
```

## 🚀 Getting Started

### Requirements
- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- matplotlib
- seaborn
- customtkinter
- openpyxl (for Excel support)

### Installation

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn customtkinter openpyxl
```

### Running the Application

**As Python Script:**
```bash
python interface.py
```

**As Compiled Desktop App:**
```bash
dist\interface\interface.exe
```

## 📊 Supported Analyses

The application provides a modular interface with dedicated sections for:
- **PCA Analysis** - Inertia, factorial planes, correlation circles
- **AFC Analysis** - Chi-square testing, frequency analysis, factorial planes
- **Clustering** - K-means with k=3,4,5, percentage distribution
- **Machine Learning** - Random Forest metrics, predictions on new data
- **Cybersecurity Anomaly Detection** - Risk assessment using ensemble methods

## 💾 Output

All analysis results are automatically exported to the `exports/` folder in CSV format for further processing or reporting.

## 🔧 Technologies Used

- **Data Analysis**: pandas, NumPy, SciPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **GUI Framework**: CustomTkinter
- **Packaging**: PyInstaller
- **Data Format**: Excel (.xlsx), CSV
