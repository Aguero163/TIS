#!/usr/bin/env python3
"""
Start the Telematics Insurance API Server on port 5001
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Starting Telematics Insurance API Server on port 5001...")

    # Change to the src/api directory
    api_dir = Path(__file__).parent.parent / 'src' / 'api'
    os.chdir(api_dir)

    # Set environment variables
    os.environ['FLASK_APP'] = 'api_server.py'
    os.environ['FLASK_ENV'] = 'development'

    print("API Server will be available at: http://localhost:5001")
    print("API Documentation: http://localhost:5001/api/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Modify the API server to use port 5001 on the fly
        with open('api_server.py', 'r') as f:
            content = f.read()

        # Replace port 5000 with 5001
        modified_content = content.replace('port=5000', 'port=5001')

        # Write temporary file
        with open('api_server_5001.py', 'w') as f:
            f.write(modified_content)

        # Run the modified server
        subprocess.run([sys.executable, 'api_server_5001.py'], check=True)
    except KeyboardInterrupt:
        print("\nAPI Server stopped by user")
    except Exception as e:
        print(f"Error starting API server: {e}")
    finally:
        # Clean up temporary file
        try:
            os.remove('api_server_5001.py')
        except:
            pass

if __name__ == "__main__":
    main()
