# 📱 Building Android APK - Concentration Tracker

## Prerequisites

### Option 1: Using Linux (Recommended)

**Best approach:** Use Ubuntu 20.04+ (or Docker)

1. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y git zip unzip openjdk-11-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
   ```

2. **Install Buildozer:**
   ```bash
   pip3 install --user buildozer
   export PATH=$PATH:~/.local/bin
   ```

3. **Install Cython:**
   ```bash
   pip3 install --user Cython==0.29.33
   ```

### Option 2: Using Docker (Easier - Works on Mac/Windows)

1. **Install Docker Desktop**

2. **Use Buildozer Docker image:**
   ```bash
   docker pull kivy/buildozer
   ```

## Building the APK

### Step 1: Prepare Your Environment

```bash
cd /path/to/Concentration-Tracker
```

### Step 2: Install Python Dependencies

```bash
pip install kivy buildozer cython
```

### Step 3: Initialize Buildozer (if needed)

```bash
buildozer init
```

This creates `buildozer.spec` (already provided)

### Step 4: Build the APK

**For Debug APK (faster, larger):**
```bash
buildozer android debug
```

**For Release APK (smaller, optimized):**
```bash
buildozer android release
```

**Note:** First build takes 30-60 minutes as it compiles all dependencies.

### Step 5: Find Your APK

After successful build, find your APK in:
```
bin/concentrationtracker-1.0-arm64-v8a-debug.apk
```

## Important Notes

### ⚠️ Known Issues & Solutions

1. **OpenCV and MediaPipe on Android:**
   - These libraries are large and complex
   - First build will take a long time
   - APK size will be ~100-200MB

2. **Camera Permissions:**
   - Already configured in `buildozer.spec`
   - App will request permissions on first launch

3. **Sound Playback:**
   - Uses Android MediaPlayer
   - Requires `alert.wav` in app directory

### Alternative: Simplified Build

If the full build is too complex, consider:

1. **Use Chaquopy** (Python for Android plugin)
2. **Use BeeWare Briefcase** (alternative packaging)
3. **Build on cloud service** (GitHub Actions, etc.)

## Testing

1. **Enable Developer Options** on your Android device
2. **Enable USB Debugging**
3. **Connect device via USB**
4. **Install APK:**
   ```bash
   adb install bin/concentrationtracker-1.0-arm64-v8a-debug.apk
   ```

## Troubleshooting

### Build Fails with OpenCV/MediaPipe Errors

**Solution:** These libraries need special Android builds. You may need to:
- Use pre-built wheels for Android
- Modify buildozer.spec to use specific versions
- Consider using OpenCV Android SDK directly

### APK Too Large

**Solution:**
- Build separate APKs per architecture (remove `arm64-v8a, armeabi-v7a`, build one at a time)
- Use `android.release_artifact = aab` for Google Play (smaller)

### Camera Not Working

**Solution:**
- Check permissions in Android Settings
- Ensure `android.permissions = CAMERA` in buildozer.spec

## Quick Build Script

Create `build_apk.sh`:

```bash
#!/bin/bash
echo "🔨 Building Android APK..."
echo "This may take 30-60 minutes on first build"
buildozer android debug
echo "✅ Build complete! Check bin/ directory"
```

Make it executable:
```bash
chmod +x build_apk.sh
./build_apk.sh
```

## Alternative: Use Cloud Build Service

Consider using:
- **GitHub Actions** with buildozer
- **GitLab CI/CD**
- **Bitrise** or **AppCenter**

These services can build your APK automatically.

## Need Help?

If you encounter issues:
1. Check Buildozer logs: `buildozer android debug -v`
2. Check Python-for-Android issues
3. Consider using a simpler approach (web app with Capacitor)

