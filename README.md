# Multimodal Biosignal Analysis

This project focuses on **cortico-muscular coherence (CMC)** analysis during motor tasks and external stimuli. The
framework integrates real-time measurement and visualization during experiments, as well as post-processing and
in-depth statistical analysis afterwards. EEG and EMG signals are recorded externally, while ECG, force, and galvanic
skin response are simultaneously monitored from a serial connection to a custom microcontroller circuit. Music
playback is controlled programmatically as an experimental stimulus.

The code was developed as a student project at the Institute of Neuroinformatics, University of Zurich (UZH) and ETH
Zurich, and is a **WORK IN PROGRESS**.


## 1. What is the goal?
The goal of this project is to provide a comprehensive, multimodal biosignal analysis platform that enables
researchers to study the coupling (coherence) between cortical neural signals and muscular activity during motor tasks
and in response to external stimuli, while also capturing cardiac activity, physical force, and skin conductance
markers via a custom microcontroller.

The platform supports the full research lifecycle from data acquisition through statistical reporting.

## 2. How is it done?
The system implements measurement and visualization modules for real-time data acquisition and display during
experiments. EEG and EMG are recorded externally using dedicated acquisition hardware (OTB4 interface), while ECG,
force, and galvanic skin response are acquired through a serial connection to a custom Teensy microcontroller circuit.

Advanced post-processing pipelines handle OTB4 file import, multimodal data integration and time alignment,
signal preprocessing, feature extraction, and comprehensive statistical modelling (omnibus and post-hoc testing), with
a dedicated reporting pipeline for result visualization.

## 3. Why this approach?
Cortico-muscular coherence (CMC) analysis serves as a potential biomarker for motor recovery and rehabilitation,
offering key insights into the functional connectivity between motor cortex activity and muscle activation. By
quantifying the synchronization between brain and muscle signals, CMC helps in understanding cortico-muscular control
mechanisms, motor coordination, and plasticity after injury or disease.

This approach allows researchers to non-invasively probe the integrity and efficiency of motor pathways,
making it especially relevant for neurorehabilitation and motor neuroscience. Integrating real-time monitoring of
cardiac, force, and autonomic signals with externally recorded EEG/EMG improves experimental control and data richness,
supporting reliable, comprehensive analyses.

## 4. What's next?
- Conduct and finalize the statistical analysis
- Improve test coverage in `tests/`

## 5. Repository Structure

    multimodal-biosignal-analysis/
    ├── src/ # Source code
    │ ├── pipeline/ # Core pipeline classes and methods
    │ │ ├── cbpa.py # Cortico-muscular coherence pipeline
    │ │ ├── data_analysis.py # Data analysis routines
    │ │ ├── data_integration.py # Multimodal data integration & time alignment
    │ │ ├── data_surrogation.py # Surrogate data generation
    │ │ ├── heterogeneity_modelling.py # Between-subject variability modelling
    │ │ ├── measurements_and_interactive_visuals.py # Real-time acquisition & live plots
    │ │ ├── music_control.py # Music stimulus control
    │ │ ├── otb_file_handling.py # OTB4 file I/O
    │ │ ├── preprocessing.py # Signal preprocessing
    │ │ ├── sensor_calibration.py # Sensor calibration routines
    │ │ ├── serial_testing.py # Serial connection testing
    │ │ ├── signal_features.py # Feature extraction
    │ │ ├── statistical_modelling.py # Statistical tests & modelling
    │ │ ├── statistical_reporting.py # Report generation
    │ │ └── visualizations.py # Plotting utilities
    │ │ 
    │ ├── experiment_workflow.py # Experiment execution workflow
    │ │ 
    │ ├── otb4_import_workflow.py # OTB4 data import workflow
    │ ├── data_integration_workflow.py # Data integration & alignment workflow
    │ ├── preprocessing_workflow.py # Preprocessing workflow
    │ ├── feature_extraction_workflow.py # Feature extraction workflow
    │ │ 
    │ ├── descriptive_statistics_workflow.py # Descriptive statistics workflow
    │ ├── statistics_data_preparation.py # Statistics data preparation
    │ ├── statistics_RQ_A_omnibus_testing_workflow.py # Omnibus tests – RQ A
    │ ├── statistics_RQ_A_post_hoc_testing_workflow.py # Post-hoc tests – RQ A
    │ ├── statistics_RQ_B_omnibus_testing_workflow.py # Omnibus tests – RQ B
    │ ├── statistics_report_workflow.py # Statistical reporting workflow
    │ │ 
    │ └── time_alignment_validation_workflow.py # Time alignment validation
    │ 
    ├── teensy-src/ # Microcontroller firmware (Teensy)
    ├── config/ # Experiment configuration files
    ├── data/ # Input data and saved outputs
    ├── tests/ # Unit and integration tests
    └── environment.yml # Conda environment specification

## 6. How to run?
### 6.1. Required Modules
It is recommended to install all required modules by creating a conda environment:
```bash
conda env create -f environment.yml
```

### 6.2 Workflow Overview

Each stage of the analysis pipeline has a dedicated ```*_workflow.py``` script in ```src/```. Run them in the following order:

#### 6.2.1. Data Collection
- Experiment – ```src/experiment_workflow.py```: real-time data acquisition and stimulus control

#### 6.2.2. Data Preprocessing
- OTB4 Import – ```src/otb4_import_workflow.py```: import raw OTB4 EEG/EMG recordings

- Data Integration – ```src/data_integration_workflow.py```: synchronize and merge all modalities

- Time Alignment Validation – ```src/time_alignment_validation_workflow.py```: validate temporal alignment

- Preprocessing – ```src/preprocessing_workflow.py```: filter, clean, and epoch signals

#### 6.2.3. Feature Extraction
- Feature Extraction – ```src/feature_extraction_workflow.py```: compute signal features (CMC, power, etc.)

- Statistics Preparation – ```src/statistics_data_preparation.py```: prepare data for inferential tests

#### 6.2.4. Statistical Modelling
- Descriptive Statistics – ```src/descriptive_statistics_workflow.py```: explore data distributions

- Omnibus Testing (RQ A) – ```src/statistics_RQ_A_omnibus_testing_workflow.py```

- Post-hoc Testing (RQ A) – ```src/statistics_RQ_A_post_hoc_testing_workflow.py```

- Omnibus Testing (RQ B) – ```src/statistics_RQ_B_omnibus_testing_workflow.py```

#### 6.2.5. Statistical Result Export
- Statistical Reporting – ```src/statistics_report_workflow.py```: generate result visualizations and reports


## 7. Other Important Information
### 7.1. Authors and Acknowledgment
Paul Rüsing with gratefully acknowledged support from the Institute of Neuroinformatics, UZH and ETH Zurich.

### 7.2. License
The project is licensed under the MIT license. To view a copy of this license, see [LICENSE](https://github.com/paulruesing/multimodal-biosignal-analysis?tab=MIT-1-ov-file).
