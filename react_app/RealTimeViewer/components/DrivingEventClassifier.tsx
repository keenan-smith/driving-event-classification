import React, { useState, useEffect, useRef } from "react";
import { View, Text, StyleSheet, Alert, TouchableOpacity } from "react-native";
import { Magnetometer, Accelerometer, Gyroscope } from "expo-sensors";
import { InferenceSession } from "onnxruntime-react-native";
import { Asset } from "expo-asset";

interface SensorData {
	x: number;
	y: number;
	z: number;
}

interface ScalerParams {
	mean: number[];
	scale: number[];
	feature_names: string[];
}

interface ClassificationResult {
	event: string;
	confidence: number;
	timestamp: number;
}

const DrivingEventClassifier: React.FC = () => {
	const [session, setSession] = useState<InferenceSession | null>(null);
	const [scalerParams, setScalerParams] = useState<ScalerParams | null>(null);
	const [isInferenceRunning, setIsInferenceRunning] = useState(false);
	const [currentResult, setCurrentResult] =
		useState<ClassificationResult | null>(null);
	const [sensorData, setSensorData] = useState({
		acc: { x: 0, y: 0, z: 0 },
		gyro: { x: 0, y: 0, z: 0 },
		mag: { x: 0, y: 0, z: 0 },
	});

	const inferenceIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
		null
	);
	const magnetometerSubscriptionRef = useRef<any>(null);
	const accelerometerSubscriptionRef = useRef<any>(null);
	const gyroscopeSubscriptionRef = useRef<any>(null);

	// Event class labels
	const eventLabels = [
		"Safe Driving",
		"Sudden Acceleration",
		"Sudden Braking",
		"Sudden Lane Change",
		"Sudden Turn",
	];

	// Load the ONNX model and scaler parameters
	useEffect(() => {
		const loadModel = async () => {
			try {
				// Load scaler parameters
				const scalerAsset = Asset.fromModule(
					require("@/assets/scaler_params.json")
				);
				await scalerAsset.downloadAsync();
				const scalerResponse = await fetch(scalerAsset.uri);
				const scalerData: ScalerParams = await scalerResponse.json();
				setScalerParams(scalerData);

				// Load ONNX model
				const modelAsset = Asset.fromModule(
					require("@/assets/driving_model.onnx")
				);
				await modelAsset.downloadAsync();
				const modelResponse = await fetch(modelAsset.uri);
				const modelArrayBuffer = await modelResponse.arrayBuffer();

				const inferenceSession = await InferenceSession.create(
					modelArrayBuffer
				);
				setSession(inferenceSession);

				console.log("Model and scaler parameters loaded successfully");
			} catch (error) {
				console.error("Error loading model:", error);
				Alert.alert(
					"Error",
					"Failed to load the driving event classification model"
				);
			}
		};

		loadModel();

		return () => {
			if (session) {
				session.release();
			}
		};
	}, []);

	// Calculate magnitudes from sensor data
	const calculateMagnitudes = (
		acc: SensorData,
		gyro: SensorData,
		mag: SensorData
	) => {
		const accMag = Math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2);
		const gyroMag = Math.sqrt(gyro.x ** 2 + gyro.y ** 2 + gyro.z ** 2);
		const magMag = Math.sqrt(mag.x ** 2 + mag.y ** 2 + mag.z ** 2);

		return { accMag, gyroMag, magMag };
	};

	// Normalize features using the scaler parameters
	const normalizeFeatures = (features: number[]): number[] => {
		if (!scalerParams) return features;

		return features.map((feature, index) => {
			const mean = scalerParams.mean[index];
			const scale = scalerParams.scale[index];
			return (feature - mean) / scale;
		});
	};

	// Run inference on current sensor data
	const runInference = async () => {
		if (!session || !scalerParams || !isInferenceRunning) return;

		try {
			const { accMag, gyroMag, magMag } = calculateMagnitudes(
				sensorData.acc,
				sensorData.gyro,
				sensorData.mag
			);

			// Create feature vector in the same order as training data
			const features = [
				sensorData.acc.x,
				sensorData.acc.y,
				sensorData.acc.z,
				sensorData.gyro.x,
				sensorData.gyro.y,
				sensorData.gyro.z,
				sensorData.mag.x,
				sensorData.mag.y,
				sensorData.mag.z,
				accMag,
				gyroMag,
				magMag,
			];

			// Normalize features
			const normalizedFeatures = normalizeFeatures(features);

			// Prepare input tensor
			const inputTensor = new Float32Array(normalizedFeatures);
			const input = {
				data: inputTensor,
				dims: [1, normalizedFeatures.length],
			};

			// Run inference
			const feeds: Record<string, any> = {};
			feeds[session.inputNames[0]] = input;

			const results = await session.run(feeds);
			const output = results[session.outputNames[0]];

			// Get prediction
			const predictions = Array.from(output.data as Float32Array);
			const maxIndex = predictions.indexOf(Math.max(...predictions));
			const predictedEvent = eventLabels[maxIndex];
			const confidence = predictions[maxIndex];

			setCurrentResult({
				event: predictedEvent,
				confidence: confidence,
				timestamp: Date.now(),
			});
		} catch (error) {
			console.error("Inference error:", error);
		}
	};

	// Start sensor subscriptions and inference
	const startInference = () => {
		if (isInferenceRunning) return;

		// Set update intervals
		Magnetometer.setUpdateInterval(50);
		Accelerometer.setUpdateInterval(50);
		Gyroscope.setUpdateInterval(50);

		// Subscribe to sensors
		magnetometerSubscriptionRef.current = Magnetometer.addListener(
			(data) => {
				setSensorData((prev) => ({ ...prev, mag: data }));
			}
		);

		accelerometerSubscriptionRef.current = Accelerometer.addListener(
			(data) => {
				setSensorData((prev) => ({ ...prev, acc: data }));
			}
		);

		gyroscopeSubscriptionRef.current = Gyroscope.addListener((data) => {
			setSensorData((prev) => ({ ...prev, gyro: data }));
		});

		// Start inference loop
		setIsInferenceRunning(true);
		inferenceIntervalRef.current = setInterval(runInference, 50);
	};

	// Stop sensor subscriptions and inference
	const stopInference = () => {
		if (!isInferenceRunning) return;

		// Clear inference interval
		if (inferenceIntervalRef.current) {
			clearInterval(inferenceIntervalRef.current);
			inferenceIntervalRef.current = null;
		}

		// Unsubscribe from sensors
		if (magnetometerSubscriptionRef.current) {
			magnetometerSubscriptionRef.current.remove();
			magnetometerSubscriptionRef.current = null;
		}

		if (accelerometerSubscriptionRef.current) {
			accelerometerSubscriptionRef.current.remove();
			accelerometerSubscriptionRef.current = null;
		}

		if (gyroscopeSubscriptionRef.current) {
			gyroscopeSubscriptionRef.current.remove();
			gyroscopeSubscriptionRef.current = null;
		}

		setIsInferenceRunning(false);
		setCurrentResult(null);
	};

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			stopInference();
		};
	}, []);

	return (
		<View style={styles.container}>
			<Text style={styles.title}>Driving Event Classifier</Text>

			<View style={styles.statusContainer}>
				<Text style={styles.statusText}>
					Model Status: {session ? "Loaded" : "Loading..."}
				</Text>
				<Text style={styles.statusText}>
					Inference: {isInferenceRunning ? "Running" : "Stopped"}
				</Text>
			</View>

			{currentResult && (
				<View style={styles.resultContainer}>
					<Text style={styles.resultTitle}>Current Event:</Text>
					<Text style={styles.eventText}>{currentResult.event}</Text>
					<Text style={styles.confidenceText}>
						Confidence:{" "}
						{(currentResult.confidence * 100).toFixed(1)}%
					</Text>
				</View>
			)}

			<View style={styles.sensorContainer}>
				<Text style={styles.sensorTitle}>Sensor Data:</Text>
				<Text>
					Acc: X: {sensorData.acc.x.toFixed(3)} Y:{" "}
					{sensorData.acc.y.toFixed(3)} Z:{" "}
					{sensorData.acc.z.toFixed(3)}
				</Text>
				<Text>
					Gyro: X: {sensorData.gyro.x.toFixed(3)} Y:{" "}
					{sensorData.gyro.y.toFixed(3)} Z:{" "}
					{sensorData.gyro.z.toFixed(3)}
				</Text>
				<Text>
					Mag: X: {sensorData.mag.x.toFixed(3)} Y:{" "}
					{sensorData.mag.y.toFixed(3)} Z:{" "}
					{sensorData.mag.z.toFixed(3)}
				</Text>
			</View>

			<View style={styles.buttonContainer}>
				<TouchableOpacity
					style={[
						styles.button,
						isInferenceRunning
							? styles.stopButton
							: styles.startButton,
					]}
					onPress={
						isInferenceRunning ? stopInference : startInference
					}
				>
					<Text style={styles.buttonText}>
						{isInferenceRunning
							? "Stop Inference"
							: "Start Inference"}
					</Text>
				</TouchableOpacity>
			</View>
		</View>
	);
};

const styles = StyleSheet.create({
	container: {
		flex: 1,
		padding: 20,
		backgroundColor: "#f5f5f5",
	},
	title: {
		fontSize: 24,
		fontWeight: "bold",
		textAlign: "center",
		marginBottom: 20,
		color: "#333",
	},
	statusContainer: {
		backgroundColor: "#fff",
		padding: 15,
		borderRadius: 8,
		marginBottom: 15,
		elevation: 2,
		shadowColor: "#000",
		shadowOffset: { width: 0, height: 2 },
		shadowOpacity: 0.1,
		shadowRadius: 4,
	},
	statusText: {
		fontSize: 16,
		marginBottom: 5,
		color: "#666",
	},
	resultContainer: {
		backgroundColor: "#e8f5e8",
		padding: 15,
		borderRadius: 8,
		marginBottom: 15,
		elevation: 2,
		shadowColor: "#000",
		shadowOffset: { width: 0, height: 2 },
		shadowOpacity: 0.1,
		shadowRadius: 4,
	},
	resultTitle: {
		fontSize: 18,
		fontWeight: "bold",
		marginBottom: 10,
		color: "#2d5a2d",
	},
	eventText: {
		fontSize: 20,
		fontWeight: "bold",
		color: "#2d5a2d",
		marginBottom: 5,
	},
	confidenceText: {
		fontSize: 16,
		color: "#2d5a2d",
	},
	sensorContainer: {
		backgroundColor: "#fff",
		padding: 15,
		borderRadius: 8,
		marginBottom: 15,
		elevation: 2,
		shadowColor: "#000",
		shadowOffset: { width: 0, height: 2 },
		shadowOpacity: 0.1,
		shadowRadius: 4,
	},
	sensorTitle: {
		fontSize: 18,
		fontWeight: "bold",
		marginBottom: 10,
		color: "#333",
	},
	buttonContainer: {
		alignItems: "center",
	},
	button: {
		paddingHorizontal: 30,
		paddingVertical: 15,
		borderRadius: 25,
		fontSize: 18,
		fontWeight: "bold",
		textAlign: "center",
		elevation: 3,
		shadowColor: "#000",
		shadowOffset: { width: 0, height: 2 },
		shadowOpacity: 0.2,
		shadowRadius: 4,
	},
	startButton: {
		backgroundColor: "#4CAF50",
	},
	stopButton: {
		backgroundColor: "#f44336",
	},
	buttonText: {
		fontSize: 18,
		fontWeight: "bold",
		color: "#fff",
	},
});

export default DrivingEventClassifier;
