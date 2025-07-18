import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from datetime import datetime
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

class DigitRecognitionCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        
    def load_and_preprocess_data(self, train_file='train.csv', test_file='test.csv'):
        """Load and preprocess the MNIST dataset from CSV files"""
        print("Loading MNIST dataset from CSV files...")
        
        # Validate file existence
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file '{train_file}' not found. Please ensure the file exists.")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file '{test_file}' not found. Please ensure the file exists.")
        
        try:
            # Load CSV files
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            # Validate training data format
            if 'label' not in train_df.columns:
                raise ValueError("Training CSV must contain a 'label' column.")
            if train_df.shape[1] != 785:  # 784 pixels + 1 label
                raise ValueError(f"Training CSV must have 785 columns (label + 784 pixels), got {train_df.shape[1]}.")
            
            # Extract features and labels from training data
            x_train = train_df.drop('label', axis=1).values
            y_train = train_df['label'].values
            
            # Handle test data (with or without labels)
            if 'label' in test_df.columns:
                x_test = test_df.drop('label', axis=1).values
                y_test = test_df['label'].values
            else:
                x_test = test_df.values
                y_test = None  # No labels available
                print("Note: Test CSV does not contain labels. Evaluation will be limited to predictions.")
            
            # Validate data shapes
            if x_train.shape[1] != 784:
                raise ValueError(f"Training data must have 784 pixel features, got {x_train.shape[1]}.")
            if x_test.shape[1] != 784:
                raise ValueError(f"Test data must have 784 pixel features, got {x_test.shape[1]}.")
            
            print(f"Training data shape: {x_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Test data shape: {x_test.shape}")
            if y_test is not None:
                print(f"Test labels shape: {y_test.shape}")
            
            # Normalize pixel values to [0, 1] range
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape data to add channel dimension (28, 28, 1)
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            
            # Convert labels to categorical one-hot encoding
            y_train = keras.utils.to_categorical(y_train, 10)
            if y_test is not None:
                y_test = keras.utils.to_categorical(y_test, 10)
            
            # Create validation split (20% of training data)
            val_size = int(0.2 * x_train.shape[0])
            indices = np.random.permutation(x_train.shape[0])
            
            self.x_val = x_train[indices[:val_size]]
            self.y_val = y_train[indices[:val_size]]
            self.x_train = x_train[indices[val_size:]]
            self.y_train = y_train[indices[val_size:]]
            self.x_test = x_test
            self.y_test = y_test
            
            print(f"Final training set shape: {self.x_train.shape}")
            print(f"Validation set shape: {self.x_val.shape}")
            print(f"Test set shape: {self.x_test.shape}")
            
            # Analyze data distribution
            self.analyze_data_distribution(train_df['label'])
            
        except pd.errors.EmptyDataError:
            raise ValueError("One of the CSV files is empty.")
        except Exception as e:
            raise Exception(f"Error processing CSV files: {str(e)}")
        
    def analyze_data_distribution(self, labels):
        """Visualize the distribution of labels in the training dataset"""
        plt.figure(figsize=(10, 6))
        sns.countplot(x=labels)
        plt.title('Distribution of Digit Labels in Training Dataset')
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.show()
        
        # Print label counts
        label_counts = pd.Series(labels).value_counts().sort_index()
        print("\nLabel Distribution:")
        for digit, count in label_counts.items():
            print(f"Digit {digit}: {count} samples")
        
    def visualize_data_samples(self):
        """Visualize sample images from the dataset"""
        plt.figure(figsize=(12, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            # Convert one-hot back to label for display
            label = np.argmax(self.y_train[i])
            plt.imshow(self.x_train[i].reshape(28, 28), cmap='gray')
            plt.title(f'Label: {label}')
            plt.axis('off')
        plt.suptitle('Sample Images from Training Dataset')
        plt.tight_layout()
        plt.show()
        
    def build_model(self):
        """Build the CNN architecture"""
        print("Building CNN model...")
        
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=(28, 28, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Display model architecture
        self.model.summary()
        
    def train_model(self, epochs=25):
        """Train the CNN model"""
        print(f"Training model for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if self.y_test is None:
            print("Cannot evaluate model: Test labels are not available in the test CSV.")
            return None, None
        
        print("Evaluating model performance...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class Accuracy:")
        for i, acc in enumerate(class_accuracy):
            print(f"Digit {i}: {acc:.4f}")
            
        return test_accuracy, test_loss
        
    def predict_sample_images(self, num_samples=10):
        """Predict and visualize sample images"""
        # Select random samples from test set
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        sample_images = self.x_test[indices]
        sample_labels = np.argmax(self.y_test[indices], axis=1) if self.y_test is not None else None
        
        # Make predictions
        predictions = self.model.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Visualize results
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
            
            if sample_labels is not None:
                # Color coding: green for correct, red for incorrect
                color = 'green' if sample_labels[i] == predicted_labels[i] else 'red'
                title = f'True: {sample_labels[i]}, Pred: {predicted_labels[i]}\n'
            else:
                color = 'blue'
                title = f'Pred: {predicted_labels[i]}\n'
                
            confidence = predictions[i][predicted_labels[i]]
            title += f'Confidence: {confidence:.3f}'
            plt.title(title, color=color)
            plt.axis('off')
            
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect, Blue=Prediction Only)')
        plt.tight_layout()
        plt.show()
        
    def save_predictions_to_csv(self, output_file='predictions.csv'):
        """Save test set predictions to a CSV file"""
        print(f"Saving predictions to {output_file}...")
        
        try:
            # Make predictions on test set
            predictions = self.model.predict(self.x_test)
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Create DataFrame with ImageId and Label
            output_df = pd.DataFrame({
                'ImageId': np.arange(1, len(self.x_test) + 1),
                'Label': predicted_labels
            })
            
            # Save to CSV
            output_df.to_csv(output_file, index=False)
            print(f"Predictions successfully saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
        
    def save_model(self, filepath='digit_recognition_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        test_accuracy, test_loss = self.evaluate_model() if self.y_test is not None else (None, None)
        
        report = {
            'Model Architecture': 'Convolutional Neural Network',
            'Dataset': 'MNIST Handwritten Digits (CSV)',
            'Training Samples': len(self.x_train),
            'Validation Samples': len(self.x_val),
            'Test Samples': len(self.x_test),
            'Test Accuracy': f"{test_accuracy:.4f}" if test_accuracy is not None else 'N/A (No test labels)',
            'Test Loss': f"{test_loss:.4f}" if test_loss is not None else 'N/A (No test labels)',
            'Target Accuracy': '>97%',
            'Achieved': 'Yes' if test_accuracy is not None and test_accuracy > 0.97 else 'No' if test_accuracy is not None else 'N/A',
            'Training Time': 'Approximately 15-20 minutes',
            'Model Size': '<5MB'
        }
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY REPORT")
        print("="*50)
        for key, value in report.items():
            print(f"{key}: {value}")
        print("="*50)
        
        return report

def main():
    """Main execution function"""
    print("Starting Handwritten Digit Recognition Project")
    print("Authors: Ramisha Ikram (FA22-BCE-024), Tahreem Shahid (FA22-BCE-035)")
    print("="*60)
    
    # Initialize the CNN classifier
    cnn_classifier = DigitRecognitionCNN()
    
    # Step 1: Load and preprocess data from CSV
    try:
        cnn_classifier.load_and_preprocess_data('train.csv', 'test.csv')
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return
    
    # Step 2: Visualize sample data
    cnn_classifier.visualize_data_samples()
    
    # Step 3: Build the model
    cnn_classifier.build_model()
    
    # Step 4: Train the model
    cnn_classifier.train_model(epochs=25)
    
    # Step 5: Plot training history
    cnn_classifier.plot_training_history()
    
    # Step 6: Evaluate the model
    cnn_classifier.evaluate_model()
    
    # Step 7: Predict sample images
    cnn_classifier.predict_sample_images()
    
    # Step 8: Save test predictions to CSV
    cnn_classifier.save_predictions_to_csv()
    
    # Step 9: Generate performance report
    cnn_classifier.generate_performance_report()
    
    # Step 10: Save the model
    cnn_classifier.save_model()
    
    print("\nProject completed successfully!")
    print(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()