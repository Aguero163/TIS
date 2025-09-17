#!/usr/bin/env python3
"""
Train ML models for risk scoring
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.telematics_simulator import TelematicsDataSimulator
from data_processing.data_processor import TelematicsDataProcessor
from ml_models.risk_scorer import RiskScorer, ModelEvaluator

def main():
    print("Telematics Insurance - Model Training")
    print("=" * 50)

    # Check if sample data exists, generate if not
    sample_data_path = 'data/samples/sample_telematics_data.csv'

    if not os.path.exists(sample_data_path):
        print("No sample data found. Generating new data...")
        simulator = TelematicsDataSimulator()
        sample_data = simulator.generate_multiple_trips(num_drivers=20, trips_per_driver=30)
        sample_data.to_csv(sample_data_path, index=False)
        print(f"Sample data saved to {sample_data_path}")

    # Load data
    print("Loading telematics data...")
    raw_data = pd.read_csv(sample_data_path)
    print(f"Loaded {len(raw_data)} records from {raw_data['driver_id'].nunique()} drivers")

    # Process data
    print("Processing telematics data...")
    processor = TelematicsDataProcessor()
    risk_features_list = processor.batch_process_drivers(raw_data)
    features_df = processor.features_to_dataframe(risk_features_list)

    print(f"Extracted features for {len(features_df)} drivers")

    # Train models
    print("Training machine learning models...")
    risk_scorer = RiskScorer()
    results = risk_scorer.train_models(features_df)

    # Display results
    print("\nTraining Results:")
    ModelEvaluator.print_model_comparison(results)

    # Save models
    model_path = 'models/risk_scoring_models.pkl'
    risk_scorer.save_models(model_path)
    print(f"\nModels saved to {model_path}")

    # Save processed features for later use
    features_path = 'data/processed/driver_features.csv'
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
