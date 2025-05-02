# Driver Monitoring System

A real-time driver monitoring system that detects drowsiness and excessive yawning using computer vision and sensor integration. The system is designed to enhance road safety by alerting drivers during fatigue or early signs of intoxication.

---

## 🔧 Features

- 👀 **Drowsiness Detection** using Eye Aspect Ratio (EAR)
- 😮 **Yawn Detection** using Mouth Aspect Ratio (MAR)
- 🧠 Facial landmark detection using **dlib** and **OpenCV**
- 🔊 Audio alerts using **pygame.mixer**
- 🛠️ Physical alerts (buzzer + vibration motor) via **Arduino + pyFirmata**
- 📸 Real-time webcam feed analysis

---

## 🧑‍💻 Tech Stack

- **Python 3**
- **OpenCV**, **dlib**, **imutils**
- **Pygame** (for sound)
- **PyFirmata** (Arduino interface)
- **NumPy**, **SciPy**
- **Arduino UNO + Buzzer + Vibration Motor**

---

## 🖼️ System Architecture

1. **Video Stream** from webcam is analyzed frame-by-frame.
2. Facial landmarks are extracted using **dlib's 68-point predictor**.
3. **EAR** is computed to detect eye closure.
4. **MAR** is calculated to detect yawning.
5. If thresholds are breached:
   - Audio alarm is played (`alarm.wav`)
   - Arduino triggers buzzer and motor for physical alert

---

## ⚙️ Installation

### 🐍 Python Dependencies

Install required libraries:
```bash
pip install opencv-python dlib imutils pygame scipy numpy pyfirmata
```
🔌 Arduino Setup
1. Connect your buzzer to pin 7 and vibration motor to pin 5 of Arduino UNO.
2. Upload the StandardFirmata sketch to your Arduino using the Arduino IDE.
3. Confirm the correct COM port in the code (default is COM3 — change if needed).
