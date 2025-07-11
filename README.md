# psBLUP Analysis

A comparative analysis project implementing **penalized structured Best Linear Unbiased Prediction (psBLUP)** and **ridge regression BLUP (rrBLUP)** methods for genomic prediction.

## Overview

This project includes:
- **Python implementation**: `psBLUP.py`, `improved_rrblup.py`
- **R implementation**: `psBLUP-main/psBLUP.R`, `rrBLUP.R`
- **Analysis scripts**: Chickpea thousand seed weight (TSW) case study
- **Plotting utilities**: Result visualization and comparison tools

## Structure

```
├── psBLUP.py               # Main Python implementation
├── psBLUP-main/psBLUP.R   # Main R implementation  
├── chickpea_tsw_analysis*  # Chickpea analysis scripts (Python & R)
├── output_*/               # Results and outputs
└── plots/                  # Generated visualizations
```

## Reference

Based on: Bartzis et al. (2022) - psBLUP method for genomic prediction. 