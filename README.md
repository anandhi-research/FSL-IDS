# FSL-IDS: Federated Stealth Learning Intrusion Detection System

This repository provides the implementation of the Federated Stealth Learning Intrusion Detection System (FSL-IDS) designed to detect stealth attacks in IoT environments using federated learning.

## Dataset

Experiments were conducted using the IoTID20 dataset.

Download dataset:
https://sites.google.com/view/iot-network-intrusion-dataset/home

Place the dataset file inside:

data/iotid20.csv

## Installation

Install dependencies:

pip install -r requirements.txt

## Run Experiment

python experiments/run_experiment.py

## Repository Structure

FSL-IDS/
│
├── data/
├── models/
├── federated/
├── utils/
├── experiments/
│
├── config.yaml
├── requirements.txt
└── README.md

## Hardware

Experiments were conducted using NVIDIA RTX 4090 GPU and Raspberry Pi 4 edge simulation.

## License

This repository is provided for research reproducibility.
