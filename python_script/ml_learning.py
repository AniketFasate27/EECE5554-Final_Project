"""
Motor Fault Detection - Machine Learning Classification
Trains multiple models and evaluates performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

class MotorFaultClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, feature_csv):
        """Load feature dataset"""
        data = pd.read_csv(feature_csv)
        
        # Separate features and labels
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Print class distribution
        class_counts = Counter(y)
        print(f"Class distribution: {dict(class_counts)}")
        
        return X, y_encoded
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split and scale data"""
        # Check if we have enough samples for stratification
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        # Only stratify if all classes have at least 2 samples
        use_stratify = min_class_count >= 2
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y if use_stratify else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        if not use_stratify:
            print("⚠️  Stratification disabled (too few samples per class)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        print("\n" + "="*60)
        print("Training Multiple Models...")
        print("="*60 + "\n")
        
        # Model 1: Random Forest (RECOMMENDED for motor diagnostics)
        print("Training Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        # Model 2: Gradient Boosting
        print("Training Gradient Boosting...")
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)
        
        # Model 3: SVM
        print("Training SVM...")
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            probability=True  # Enable probability predictions
        )
        self.models['SVM'].fit(X_train, y_train)
        
        # Model 4: Neural Network
        print("Training Neural Network...")
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        self.models['Neural Network'].fit(X_train, y_train)
        
        # Model 5: K-Nearest Neighbors
        print("Training KNN...")
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=min(5, len(X_train)),  # Adjust k based on dataset size
            weights='distance'
        )
        self.models['KNN'].fit(X_train, y_train)
        
        print("\nAll models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60 + "\n")
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"\nClassification Report:")
            
            # Only show classes that are in the test set
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            target_names = [self.label_encoder.classes_[i] for i in unique_labels]
            
            print(classification_report(
                y_test, y_pred, 
                labels=unique_labels,
                target_names=target_names,
                zero_division=0
            ))
        
        # Find best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"ACCURACY: {results[self.best_model_name]*100:.2f}%")
        print("="*60 + "\n")
        
        return results
    
    def plot_confusion_matrix(self, X_test, y_test):
        """Plot confusion matrix for best model"""
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get unique labels present in test set
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        labels = [self.label_encoder.classes_[i] for i in unique_labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance (for tree-based models)"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[-20:]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300)
            print("Feature importance saved as 'feature_importance.png'")
            plt.show()
        else:
            print(f"{self.best_model_name} does not support feature importance")
    
    def save_model(self, filename='motor_fault_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name
        }
        joblib.dump(model_data, filename)
        print(f"\nModel saved as '{filename}'")
    
    def predict_fault(self, features):
        """Predict motor fault from features"""
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.best_model.predict(features_scaled)[0]
        
        # Get probabilities if available
        if hasattr(self.best_model, 'predict_proba'):
            prediction_proba = self.best_model.predict_proba(features_scaled)[0]
            confidence = prediction_proba[prediction] * 100
        else:
            confidence = 100.0  # If no probability available
        
        # Get label
        fault_type = self.label_encoder.inverse_transform([prediction])[0]
        
        return fault_type, confidence


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOTOR FAULT DETECTION - ML TRAINING")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = MotorFaultClassifier()
    
    # Load feature dataset
    X, y = classifier.load_data('motor_features.csv')
    
    # Check dataset size
    n_samples = X.shape[0]
    
    if n_samples < 10:
        print(f"\n⚠️  WARNING: Small dataset detected ({n_samples} samples)")
        print("   Results may not be reliable. Collect more data for better performance.")
        print("   Recommended: At least 10+ samples per fault type\n")
    
    # Prepare data (with automatic stratification handling)
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.2)
    
    # Train models
    classifier.train_models(X_train, y_train)
    
    # Evaluate models
    results = classifier.evaluate_models(X_test, y_test)
    
    # Plot results
    try:
        classifier.plot_confusion_matrix(X_test, y_test)
        classifier.plot_feature_importance()
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Save best model
    classifier.save_model('motor_fault_detector.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Collect more data samples (recommended: 10+ per fault type)")
    print("2. Run this script again with more data")
    print("3. Use 'motor_fault_detector.pkl' for real-time predictions")
    print("="*60 + "\n")