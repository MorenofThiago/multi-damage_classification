# multi-damage_classification
This repository provides code for a methodology that classifies multi-damage scenarios in railway bridges using drive-by measurements. Below is presented a step-by-step guide to the main scripts and dataset organization.

### Scripts:

data-processing.py – The main script for loading, preprocessing, and analyzing the acceleration data.

hyperparameters-optimization.py – Optimizes the hyperparameters of the classification model to improve performance.

### Datasets:

The acceleration datasets are divided into 20 files, categorized by the sensor placement:

Car Body (CB): Acceleration data measured from the vehicle's car body.

Front Bogie (FB): Acceleration data measured from the vehicle's front bogie.

For each sensor position, there are 10 structural scenarios:

Baseline – Undamaged structure.

Case2 to Case10 – Different types of structural damage.

### File Naming Convention:

Data_{SensorPosition}_{Scenario}.mat
