#!/bin/bash

echo "🐳 Building Android APK using Docker"
echo "====================================="
echo ""
echo "⚠️  Note: Permission errors are harmless warnings"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "📦 Starting build (this may take 30-60+ minutes)..."
echo "   Permission warnings can be ignored."
echo ""

# Build using Docker - ignore permission warnings
docker run --rm -it \
    -v "$(pwd)":/home/user/hostcwd \
    -w /home/user/hostcwd \
    kivy/buildozer \
    buildozer android debug 2>&1 | grep -v "chown: changing ownership" | grep -v "Operation not permitted"

BUILD_EXIT_CODE=${PIPESTATUS[0]}

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    APK_FILE=$(ls -t bin/*.apk 2>/dev/null | head -1)
    if [ -n "$APK_FILE" ]; then
        echo "📱 APK location: $APK_FILE"
        echo "📦 File size: $(du -h "$APK_FILE" | cut -f1)"
    else
        echo "⚠️  APK file not found in bin/ directory"
    fi
else
    echo ""
    echo "❌ Build failed with exit code $BUILD_EXIT_CODE"
    echo "   Check the logs above for actual errors (ignore permission warnings)"
    exit 1
fi

