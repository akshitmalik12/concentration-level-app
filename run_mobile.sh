#!/bin/bash

echo "🚀 Starting Concentration Tracker for Mobile"
echo "=============================================="
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ Flask is not installed!"
    echo "📦 Installing Flask..."
    pip install flask
    echo ""
fi

# Check if port 5000 is in use and kill it
if lsof -ti:5000 > /dev/null 2>&1; then
    echo "⚠️  Port 5000 is in use. Cleaning up..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 1
fi

echo "✅ Starting server (will find available port automatically)..."
echo ""

python3 app_web.py

