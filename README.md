# Parkinson's Disease Tremor Detection using STM32 and Embedded AI

## Overview

This project implements a real-time Parkinson's Disease tremor detection system using an STM32F401RE microcontroller and an MPU9255 accelerometer. Accelerometer data is processed on the microcontroller, statistical features are extracted, and an embedded neural network deployed using STM32 X-CUBE-AI classifies tremor severity into four categories.

The system performs inference entirely on the embedded device without requiring cloud connectivity.

---

## Features

- Real-time tremor monitoring
- Embedded AI inference using STM32 X-CUBE-AI
- MPU9255 accelerometer over SPI
- UART serial output
- Sliding window feature extraction
- 13 statistical features
- Four-class tremor classification
- Offline model training using TensorFlow
- TensorFlow Lite conversion for deployment

---

## Classification Classes

- Normal
- Mild
- Moderate
- Severe

---

## Hardware Used

- STM32 Nucleo-F401RE
- MPU9255 Accelerometer
- USB Cable
- Breadboard
- Jumper Wires

---

## Software Used

- STM32CubeIDE
- STM32CubeMX
- STM32 X-CUBE-AI
- Python
- TensorFlow
- NumPy
- Pandas

---

## System Architecture

<img width="257" height="455" alt="image" src="https://github.com/user-attachments/assets/eddb6adf-5e36-4dc8-859c-10bc948ad01b" />

---

## Feature Extraction

The embedded firmware extracts 13 statistical features from a 1-second sliding window:

- Mean X
- Mean Y
- Mean Z
- Standard Deviation X
- Standard Deviation Y
- Standard Deviation Z
- Skewness X
- Skewness Y
- Skewness Z
- Mean Magnitude
- Standard Deviation Magnitude
- Maximum Magnitude
- Median Absolute Deviation

---

## Machine Learning Pipeline

<img width="255" height="455" alt="image" src="https://github.com/user-attachments/assets/1380e37c-6b25-4e56-9819-2dcf399b1a0a" />


---

## Hardware Connections

| MPU9255 | STM32F401RE |
|----------|-------------|
| VCC | 3.3V |
| GND | GND |
| SCLK | PA5 |
| MISO | PA6 |
| MOSI | PA7 |
| CS | PA4 |
| INT | PA0 (Optional) |

---
## GUIDE
See STM_GUIDE for Running

---

## Sample Output

```
Pred: NORMAL

Pred: MILD

Pred: MODERATE

Pred: SEVERE
```

---

## Future Improvements

- Bluetooth monitoring
- OLED display
- Cloud dashboard
- Mobile application
- Larger Parkinson's dataset
- TinyML optimization
- Power optimization

---

## Author

Harshini Subbiah

--

## Contributors
Varsha V

Akshaya H

Subiksha

