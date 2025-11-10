#!/bin/bash

echo "🚀 Starting Concentration Tracker for Mobile"
echo "=============================================="
echo ""

# Get IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    IP=$(hostname -I | awk '{print $1}')
else
    IP="YOUR_IP_HERE"
fi

echo "📱 To access on your phone:"
echo "1. Make sure your phone is on the same WiFi network"
echo "2. Open browser on phone and go to:"
echo ""
echo "   http://$IP:5000"
echo ""
echo "3. Allow camera access when prompted"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "Starting server..."
echo ""

python3 app_web.py

