# Run this separately to verify the saved model
import joblib
model_data = joblib.load('motor_fault_detector.pkl')
print(f"Loaded model: {model_data['model_name']}")
print(f"Classes: {model_data['label_encoder'].classes_}")