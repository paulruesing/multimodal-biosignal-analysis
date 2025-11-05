# Multimodal Biosignal Analysis

This project focuses on cortico-muscular coherence analysis during motor tasks and external stimuli. The framework
integrates real-time measurement and visualization during experiments as well as post-processing and in-depth analysis
afterwards. EEG and EMG signals are recorded externally, while ECG, force, and galvanic skin response are simultaneously
monitored from a serial connection to a custom microcontroller circuit.

The code was developed as a student project at the Institute of Neuroinformatics, University of Zurich (UZH) and ETH
Zurich, and is a **WORK IN PROGRESS**.


## 1. What is the goal?
The goal of this project is to provide a comprehensive, multimodal biosignal analysis platform that enables researchers
to study the coupling (coherence) between cortical neural signals and muscular activity during motor tasks and in
response to external stimuli, while also capturing cardiac activity, physical force, and skin conductance markers
monitored via a custom microcontroller.

## 2. How is it done?
The system implements measurement and visualization modules for real-time data acquisition and display during
experiments. EEG and EMG are recorded externally using dedicated acquisition hardware, while ECG, force, and galvanic
skin response are acquired in real-time through a serial connection to a custom microcontroller circuit. Advanced
post-processing pipelines enable synchronization, signal cleaning, and multi-modal coherence analysis.

## 3. Why this approach?
Cortico-muscular coherence (CMC) analysis serves as a potential biomarker for motor recovery and rehabilitation,
offering key insights into the functional connectivity between motor cortex activity and muscle activation. By
quantifying the synchronization between brain and muscle signals, CMC helps in understanding cortico-muscular control
mechanisms, motor coordination, and plasticity after injury or disease. This approach allows researchers to
non-invasively probe the integrity and efficiency of motor pathways, making it especially relevant for
neurorehabilitation and motor neuroscience. Integrating real-time monitoring of cardiac, force, and autonomic signals
with externally recorded EEG/EMG improves experimental control and data richness, supporting reliable, comprehensive analyses.


## 4. What's next?
- Design the experiment application
- Build the CMC computation pipeline
- Conduct the statistical analysis

## 5. Notebooks structure
- *src/*: source code directory containing classes and methods
- *notebooks/*: jupyter notebooks demonstrating the workflow and necessary for development
- *literature/*: a selection of papers explicating some theoretical underlinings
- *data/*: input data and saved models

## 6. How to run?
### 6.1. Required Modules
It is recommended to install all required modules by creating a conda environment through running
`conda env create -f environment.yml`
in terminal in the project directory.

### 6.2. Recommendations
Usage is extensively demonstrated in the notebooks, and it is advised to follow such procedure when implementing.

## 7. Other Important Information
### 7.1. Authors and Acknowledgment
Paul Rüsing with gratefully acknowledged support from the Institute of Neuroinformatics, UZH and ETH Zurich.

### 5.2. License
The project is licensed under the MIT license. To view a copy of this license, see [LICENSE](https://github.com/paulruesing/lrp-xai-pytorch?tab=MIT-1-ov-file).
