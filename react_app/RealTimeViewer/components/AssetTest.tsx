import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet } from "react-native";
import { Asset } from "expo-asset";

const AssetTest: React.FC = () => {
	const [testResults, setTestResults] = useState<string[]>([]);

	useEffect(() => {
		const testAssets = async () => {
			const results: string[] = [];

			try {
				// Test 1: Try to require the JSON file directly
				results.push("Test 1: Direct require of JSON file");
				try {
					const jsonData = require("@/assets/scaler_params.json");
					results.push("✓ JSON file loaded directly");
					results.push(
						`JSON keys: ${Object.keys(jsonData).join(", ")}`
					);
				} catch (error) {
					results.push(`✗ JSON direct load failed: ${error}`);
				}

				// Test 2: Try Asset.fromModule for JSON
				results.push("\nTest 2: Asset.fromModule for JSON");
				try {
					const scalerAsset = Asset.fromModule(
						require("@/assets/scaler_params.json")
					);
					results.push("✓ Asset.fromModule created for JSON");
					results.push(`Asset URI: ${scalerAsset.uri}`);
					await scalerAsset.downloadAsync();
					results.push("✓ Asset downloaded");
				} catch (error) {
					results.push(`✗ Asset.fromModule failed: ${error}`);
				}

				// Test 3: Try to require the ONNX file
				results.push("\nTest 3: Direct require of ONNX file");
				try {
					const onnxData = require("@/assets/driving_model.onnx");
					results.push("✓ ONNX file loaded directly");
				} catch (error) {
					results.push(`✗ ONNX direct load failed: ${error}`);
				}

				// Test 4: Try Asset.fromModule for ONNX
				results.push("\nTest 4: Asset.fromModule for ONNX");
				try {
					const modelAsset = Asset.fromModule(
						require("@/assets/driving_model.onnx")
					);
					results.push("✓ Asset.fromModule created for ONNX");
					results.push(`Asset URI: ${modelAsset.uri}`);
					await modelAsset.downloadAsync();
					results.push("✓ Asset downloaded");
				} catch (error) {
					results.push(`✗ Asset.fromModule failed: ${error}`);
				}
			} catch (error) {
				results.push(`General error: ${error}`);
			}

			setTestResults(results);
		};

		testAssets();
	}, []);

	return (
		<View style={styles.container}>
			<Text style={styles.title}>Asset Loading Test</Text>
			{testResults.map((result, index) => (
				<Text key={index} style={styles.resultText}>
					{result}
				</Text>
			))}
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
		fontSize: 20,
		fontWeight: "bold",
		marginBottom: 15,
		color: "#333",
	},
	resultText: {
		fontSize: 14,
		marginBottom: 5,
		color: "#666",
		fontFamily: "monospace",
	},
});

export default AssetTest;
