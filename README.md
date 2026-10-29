# TERA NextPM + ESP32 Real-Time Air Quality & Machine Learning Dashboards

An educational, interactive Python and C++ suite for streaming multi-channel particulate matter counts from a TERA NextPM sensor via an ESP32 microcontroller, coupled with real-time machine learning architectures built on CustomTkinter, Scikit-Learn, and Matplotlib.

## Hardware Architecture & Wiring

<img src="Hardware.jpg" alt="Hardware Setup" width="400">

### Components

* TERA NextPM Particle Sensor (UART Interface)
* ESP32 Microcontroller Board (NodeMCU / ESP32 WROOM-32)
* Jumper Wires & 5V Power Supply

### Pin Connection Table

| TERA NextPM Pin | ESP32 Pin | Note | 
| --- | --- | --- | 
| VCC | 5V / VIN | 5V Power supply | 
| GND | GND | Common Ground | 
| TX | GPIO 16 (RX2) | HardwareSerial2 RX | 
| RX | GPIO 17 (TX2) | HardwareSerial2 TX | 

## Repository Structure

```text
tera-nextpm-ml-dashboard/
├── README.md                              # Documentation and Guide
├── firmware/
│   └── NextPM_ESP32_Firmware.ino          # ESP32 C++ Arduino Sketch
└── dashboards/
    ├── NextPS_Dashboard.py                # Basic Real-Time Telemetry Stream
    ├── Gaussian_Mixture_Models.py         # Unsupervised GMM + 3D PCA Clustering
    ├── NextPS_Forecast_Dashboard.py       # Short-Term Linear Regression Forecast
    └── NextPS_Supervised3D_Dashboard.py   # Supervised Activity Recognition (Random Forest)
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/tera-nextpm-ml-dashboard.git
cd tera-nextpm-ml-dashboard
```

Replace `your-username` with your actual GitHub username.

### 2. Install Python Dependencies

```bash
pip install customtkinter pandas numpy scikit-learn matplotlib pyserial
```

### 3. Flash the ESP32 Firmware

1. Open `firmware/NextPM_ESP32_Firmware.ino` in the Arduino IDE.
2. Select your ESP32 board and the appropriate COM port.
3. Upload the sketch.

The ESP32 samples the NextPM sensor every 2 seconds and communicates at 115,200 baud using an 8E1 serial configuration (8 data bits, even parity, 1 stop bit).

## Application Scripts & Demos

### 1. Basic Real-Time Telemetry Dashboard

**Script:** `dashboards/NextPS_Dashboard.py`

**Description:** Displays live particle counts across all five particle channels (>0.3 µm, >0.5 µm, >1.0 µm, >2.5 µm, and >5.0 µm) in a rolling 60-second window. Includes descriptive statistics tools and raw CSV export.

### 2. Unsupervised GMM + 3D PCA Clustering

**Script:** `dashboards/Gaussian_Mixture_Models.py`

**Description:** Analyses multi-channel particle size ratios in real time using Gaussian Mixture Models (GMM) and the Bayesian Information Criterion (BIC) to explore potential indoor air quality regimes without manual labelling. Results are visualised in an interactive, rotatable 3D PCA space.

### 3. Predictive Forecasting & Trajectory Space

**Script:** `dashboards/NextPS_Forecast_Dashboard.py`

**Description:** Applies rolling least-squares linear regression to project particle-count trajectories 10 seconds into the future, displayed as dashed lines. Tracks temporal changes and plots a continuous 3D phase-space trajectory.

### 4. User-Defined Supervised Activity Recognition

**Script:** `dashboards/NextPS_Supervised3D_Dashboard.py`

**Description:** Allows users to enter custom activity labels (e.g., "Cooking", "Cleaning", or "Idle"), record live training samples in RAM, and train a Random Forest classifier. The dashboard predicts activity classes in real time, displays confidence scores, and colour-codes custom classes in 3D PCA space.

## Educational Guides & Scientific Background

Each advanced dashboard includes a built-in educational guide explaining the underlying concepts.

- **Particulate Matter Physics:** Discusses how particle counts across different size thresholds may help distinguish between particle-generating activities, such as cooking, and sources such as dust or movement. These relationships are context-dependent and are not definitive source identifications.

- **Dimensionality Reduction:** Explains how Principal Component Analysis (PCA) reduces a six-dimensional feature vector into three principal components for visualisation:

  `[PM0.3, PM0.5, PM1.0, PM2.5, PM5.0, Ratio]`

  The resulting axes are PC1, PC2, and PC3.

- **Machine Learning Fundamentals:** Introduces Random Forest classification, Gaussian Mixture Model covariance structures, Bayesian Information Criterion (BIC), and linear regression.

## License

Distributed under the MIT License. See `LICENSE` for more information.
