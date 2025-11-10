#!/bin/bash

echo "🔨 Building Android APK for Concentration Tracker"
echo "=================================================="
echo ""
echo "⚠️  Note: First build takes 30-60 minutes"
echo "⚠️  Requires Linux or Docker"
echo ""

# Check if buildozer is installed
if ! command -v buildozer &> /dev/null; then
    echo "❌ Buildozer not found!"
    echo "Install with: pip install buildozer"
    exit 1
fi

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: Not on Linux. Buildozer works best on Linux."
    echo "Consider using Docker: docker pull kivy/buildozer"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🚀 Starting build..."
echo ""

# Build debug APK
buildozer android debug

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "📱 APK location: bin/concentrationtracker-1.0-*-debug.apk"
    echo ""
    echo "To install on device:"
    echo "  adb install bin/concentrationtracker-1.0-*-debug.apk"
else
    echo ""
    echo "❌ Build failed. Check logs above."
    exit 1
fi

