# 📱 How to Get Your Android APK

## 🎯 Easiest Method: GitHub Actions (Recommended)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Android build configuration"
git push origin main
```

### Step 2: Wait for Build

1. Go to your GitHub repository: `https://github.com/akshitmalik12/Concenration-Tracker`
2. Click on the **Actions** tab
3. You'll see a workflow run called "Build Android APK"
4. Wait for it to complete (30-60 minutes for first build)

### Step 3: Download APK

**Option A: From Artifacts (Easiest)**
1. Click on the completed workflow run
2. Scroll down to "Artifacts"
3. Click on "concentration-tracker-apk"
4. Download the APK file

**Option B: From Releases**
1. Go to the **Releases** section of your repository
2. Download the latest release APK

## 🐳 Alternative: Docker Build (Local)

If you want to build locally (may have permission warnings - they're harmless):

```bash
./docker_build.sh
```

The APK will be in the `bin/` directory.

**Note:** Permission errors are just warnings and won't stop the build.

## ⚠️ Important Notes

1. **First build takes 30-60+ minutes** - be patient
2. **APK size will be 150-300MB** - large due to OpenCV/MediaPipe
3. **OpenCV/MediaPipe may need special handling** - if build fails, we may need custom recipes

## 🚀 Quick Start

```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for Android build"
git push

# 2. Go to GitHub Actions tab
# 3. Wait for build
# 4. Download APK from artifacts
```

That's it! The APK will be available for download from GitHub Actions.

