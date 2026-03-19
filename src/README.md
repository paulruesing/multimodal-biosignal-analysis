# Workflow Scripts
## Structure

    src/ # Source code
    ├── pipeline/ # Core pipeline classes and methods
    │ ├── cbpa.py # Cortico-muscular coherence pipeline
    │ ├── data_analysis.py # Data analysis routines
    │ ├── data_integration.py # Multimodal data integration & time alignment
    │ ├── data_surrogation.py # Surrogate data generation
    │ ├── heterogeneity_modelling.py # Between-subject variability modelling
    │ ├── measurements_and_interactive_visuals.py # Real-time acquisition & live plots
    │ ├── music_control.py # Music stimulus control
    │ ├── otb_file_handling.py # OTB4 file I/O
    │ ├── preprocessing.py # Signal preprocessing
    │ ├── sensor_calibration.py # Sensor calibration routines
    │ ├── serial_testing.py # Serial connection testing
    │ ├── signal_features.py # Feature extraction
    │ ├── statistical_modelling.py # Statistical tests & modelling
    │ ├── statistical_reporting.py # Report generation
    │ └── visualizations.py # Plotting utilities
    │ 
    ├── experiment_workflow.py # Experiment execution workflow
    │ 
    ├── otb4_import_workflow.py # OTB4 data import workflow
    ├── data_integration_workflow.py # Data integration & alignment workflow
    ├── preprocessing_workflow.py # Preprocessing workflow
    ├── feature_extraction_workflow.py # Feature extraction workflow
    │ 
    ├── descriptive_statistics_workflow.py # Descriptive statistics workflow
    ├── statistics_data_preparation.py # Statistics data preparation
    ├── statistics_RQ_A_omnibus_testing_workflow.py # Omnibus tests – RQ A
    ├── statistics_RQ_A_post_hoc_testing_workflow.py # Post-hoc tests – RQ A
    ├── statistics_RQ_B_omnibus_testing_workflow.py # Omnibus tests – RQ B
    ├── statistics_report_workflow.py # Statistical reporting workflow
    │ 
    └── time_alignment_validation_workflow.py # Time alignment validation


### Workflow Overview

Each stage of the analysis pipeline has a dedicated ```*_workflow.py``` script. Run them in the following order:

#### 1. Data Collection
- Experiment – ```src/experiment_workflow.py```: real-time data acquisition and stimulus control

#### 2. Data Preprocessing
- OTB4 Import – ```src/otb4_import_workflow.py```: import raw OTB4 EEG/EMG recordings

- Data Integration – ```src/data_integration_workflow.py```: synchronize and merge all modalities

- Time Alignment Validation – ```src/time_alignment_validation_workflow.py```: validate temporal alignment

- Preprocessing – ```src/preprocessing_workflow.py```: filter, clean, and epoch signals

#### 3. Feature Extraction
- Feature Extraction – ```src/feature_extraction_workflow.py```: compute signal features (CMC, power, etc.)

- Statistics Preparation – ```src/statistics_data_preparation.py```: prepare data for inferential tests

#### 4. Statistical Modelling
- Descriptive Statistics – ```src/descriptive_statistics_workflow.py```: explore data distributions

- Omnibus Testing (RQ A) – ```src/statistics_RQ_A_omnibus_testing_workflow.py```

- Post-hoc Testing (RQ A) – ```src/statistics_RQ_A_post_hoc_testing_workflow.py```

- Omnibus Testing (RQ B) – ```src/statistics_RQ_B_omnibus_testing_workflow.py```

#### 5. Statistical Result Export
- Statistical Reporting – ```src/statistics_report_workflow.py```: generate result visualizations and reports
