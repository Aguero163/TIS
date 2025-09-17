#!/usr/bin/env python3
"""
Generate sample telematics data for testing
"""
import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.telematics_simulator import TelematicsDataSimulator

def main():
    parser = argparse.ArgumentParser(description='Generate sample telematics data')
    parser.add_argument('--drivers', type=int, default=50, help='Number of drivers (default: 50)')
    parser.add_argument('--trips', type=int, default=30, help='Number of trips per driver (default: 30)')
    parser.add_argument('--output', type=str, default='data/samples/telematics_data.csv', help='Output file path')

    args = parser.parse_args()

    print(f"Generating telematics data for {args.drivers} drivers...")
    print(f"Each driver will have {args.trips} trips")

    # Initialize simulator
    simulator = TelematicsDataSimulator()

    # Generate data
    data = simulator.generate_multiple_trips(
        num_drivers=args.drivers,
        trips_per_driver=args.trips
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save data
    data.to_csv(args.output, index=False)

    print(f"\nData generation completed!")
    print(f"Total records: {len(data):,}")
    print(f"Total trips: {data['trip_id'].nunique():,}")
    print(f"Unique drivers: {data['driver_id'].nunique():,}")
    print(f"Output saved to: {args.output}")

    # Display sample statistics
    print("\nSample Statistics:")
    print(f"Average speed: {data['speed_mph'].mean():.1f} mph")
    print(f"Harsh braking events: {data['harsh_braking'].sum():,}")
    print(f"Harsh acceleration events: {data['harsh_acceleration'].sum():,}")
    print(f"Phone usage events: {data['phone_usage'].sum():,}")

    # Show driver profile distribution
    print("\nDriver Profile Distribution:")
    profile_counts = data.groupby('driver_id').first()['trip_id'].count()
    print(f"Data spans from {data['timestamp'].min()} to {data['timestamp'].max()}")

if __name__ == "__main__":
    main()
