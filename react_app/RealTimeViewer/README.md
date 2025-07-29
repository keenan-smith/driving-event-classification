# Real-Time Driving Event Classifier

This React Native app uses ONNX inference to classify driving events in real-time using smartphone sensor data.

## Features

-   **Real-time sensor data collection** from accelerometer, gyroscope, and magnetometer
-   **ONNX model inference** for driving event classification
-   **Live classification results** showing current driving event and confidence
-   **50ms inference interval** for responsive real-time analysis

## Supported Driving Events

The app can classify the following driving events:

1. **Safe Driving** - Normal driving behavior
2. **Sudden Acceleration** - Rapid increase in speed
3. **Sudden Braking** - Rapid deceleration
4. **Sudden Lane Change** - Abrupt lateral movement
5. **Sudden Turn** - Sharp directional changes

## How to Use

1. **Start the app** - The model will automatically load on startup
2. **Grant sensor permissions** - Allow access to device sensors when prompted
3. **Press "Start Inference"** - Begin real-time classification
4. **Monitor results** - View current driving event and confidence level
5. **Press "Stop Inference"** - Stop the classification process

## Technical Details

### Sensor Data Processing

-   Collects accelerometer, gyroscope, and magnetometer readings
-   Calculates magnitude features for orientation-independent analysis
-   Normalizes features using pre-trained scaler parameters

### Model Architecture

-   Uses a Random Forest classifier exported to ONNX format
-   Processes 12 features: 9 raw sensor values + 3 magnitude features
-   Assumes paved road conditions for consistent classification

### Performance

-   Runs inference every 50ms for real-time responsiveness
-   Updates sensor data at 50ms intervals
-   Displays confidence scores for classification reliability

## Requirements

-   React Native with Expo
-   Device with accelerometer, gyroscope, and magnetometer sensors
-   ONNX Runtime for React Native

## Files

-   `components/DrivingEventClassifier.tsx` - Main classification component
-   `assets/driving_model.onnx` - Trained ONNX model
-   `assets/scaler_params.json` - Feature normalization parameters

## Development

To run the app:

```bash
cd react_app/RealTimeViewer
npm install
npm start
```

The app will open in Expo Go or your preferred development environment.
