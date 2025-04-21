
# Traffic Accident Hotspot Prediction using Graph Neural Networks

This project explores the prediction of traffic accident hotspots using Graph Neural Networks (GCNs), applied to datasets from both **Singapore** and the **United States**. It integrates spatial, temporal, and environmental data to construct node-level features for road networks and trains deep learning models to identify high-risk zones.

## 📁 Project Structure

```
traffic-accident-hotspots/
├── README.md
├── requirements.txt
│
├── Singapore/
│   ├── data/
│   │   ├── raw/                     # Original datasets (rainfall, traffic, accident records)
│   │   └── processed/               # Cleaned data
│   │   └── data_collection.ipynb    # Notebook for loading and preprocessing data
│   ├── models/
│   │   └── hotspot predictions with SG.ipynb # Model for SG Traffic Hotspot predictions
│   ├── utils/
│   │   └── sg_node_features.ipynb   # Creating node features for Singapore 
│
├── US/
│   ├── data/
│   │   ├── TAP-city/  # US cities data
│   │   └── TAP-state/
│   ├── models/
│   │   └── hotspot predictions with US.ipynb  # Model for US Traffic Hotspot predictions
│   ├── utils/
│   │   └── load_dataset.ipynb  #  View TAP data
│   ├── output/ 
│   │  


```

## 🌏 Project Summary

- **Singapore**: Limited by small geographical scale and sparse accident data. Attempted to integrate rainfall, traffic volume, and accident statistics into a spatial-temporal GCN, but hotspot prediction was challenging due to sparse and aggregated data.
- **United States**: Switched to US datasets with richer data and larger spatial topology. Utilized TAP dataset for city and state-level prediction experiments.

## 🧠 Methods

- Spatial joins with geospatial road networks
- Node feature creation from rainfall, traffic volume, and accident records
- GCN training and evaluation using PyTorch Geometric
- Hotspot prediction and visualization

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```


## 📬 Contact

For any questions or contributions, feel free to open an issue or reach out!

