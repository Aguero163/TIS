#!/usr/bin/env python3
"""
Start the Telematics Insurance Dashboard (pointing to API on port 5001)
"""
import os
import sys
import webbrowser
import http.server
import socketserver
from pathlib import Path
import threading
import time

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent.parent / 'src' / 'dashboard'), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def modify_dashboard_for_port_5001():
    """Modify dashboard HTML to point to port 5001"""
    dashboard_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'index.html'

    try:
        with open(dashboard_path, 'r') as f:
            content = f.read()

        # Replace API URL to point to port 5001
        modified_content = content.replace(
            'const API_BASE_URL = \'http://localhost:5000/api\';',
            'const API_BASE_URL = \'http://localhost:5001/api\';'
        )

        # Write to temporary file
        temp_dashboard = Path(__file__).parent.parent / 'src' / 'dashboard' / 'index_5001.html'
        with open(temp_dashboard, 'w') as f:
            f.write(modified_content)

        return temp_dashboard
    except Exception as e:
        print(f"Warning: Could not modify dashboard: {e}")
        return dashboard_path

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:8080/index_5001.html')

def main():
    print("Starting Telematics Insurance Dashboard (API on port 5001)...")

    # Modify dashboard to point to port 5001
    modify_dashboard_for_port_5001()

    PORT = 8080

    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    print(f"Dashboard will be available at: http://localhost:{PORT}/index_5001.html")
    print("\nMake sure the API server is running on port 5001")
    print("Default login: admin / admin123")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)

    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print(f"Dashboard server started on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped by user")
    except Exception as e:
        print(f"Error starting dashboard server: {e}")

if __name__ == "__main__":
    main()
