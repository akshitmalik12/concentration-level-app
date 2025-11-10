# Build Fixes and Solutions

## Known Issues and Solutions

### 1. OpenCV and MediaPipe Requirements

**Problem**: OpenCV (`opencv-python-headless`) and MediaPipe don't have standard Python-for-Android recipes in Buildozer. They require custom recipes or alternative approaches.

**Solutions**:

#### Option A: Use Custom Recipes (Recommended for Production)
1. Create custom Python-for-Android recipes for OpenCV and MediaPipe
2. Place recipes in a `recipes/` directory
3. Configure buildozer.spec to use custom recipes:
   ```
   p4a.local_recipes = ./recipes
   ```

#### Option B: Use Pre-built Libraries
1. Download pre-built OpenCV Android libraries
2. Include them in the APK manually
3. Modify the code to use the pre-built libraries

#### Option C: Alternative Libraries (For Testing)
1. Temporarily remove OpenCV/MediaPipe from requirements
2. Use alternative face detection libraries that have Android support
3. Or create a mock version for testing the build process

### 2. Android API Configuration

**Fixed**: Set `android.api = 33` and `android.target_api = 33` in buildozer.spec
- Minimum SDK: 21 (Android 5.0)
- Target SDK: 33 (Android 13)
- NDK: 25b

### 3. Sound File Handling

**Fixed**: Improved Android sound playback to:
- Try loading from Android assets first
- Fallback to file system paths
- Handle both absolute and relative paths
- Proper URI handling for MediaPlayer

### 4. Build Dependencies

**Fixed**: Added all required system dependencies:
- Java 17 JDK
- Build tools (cmake, autoconf, libtool, etc.)
- Image libraries (libjpeg, libpng, libtiff)
- Python development headers
- Caching tools (ccache)

### 5. Error Diagnostics

**Added**: Comprehensive error extraction and reporting:
- Recipe/requirements errors
- OpenCV/MediaPipe specific errors
- Compilation errors
- Last 50 lines of build log
- Detailed error summary in workflow

## Next Steps

1. **If build fails due to OpenCV/MediaPipe**:
   - Check the build-log artifact for specific errors
   - Consider using Option C (alternative libraries) for initial testing
   - Then implement Option A (custom recipes) for production

2. **If build succeeds**:
   - Test the APK on an Android device
   - Verify camera permissions
   - Test sound playback
   - Verify face detection works

3. **Custom Recipe Creation**:
   - See Python-for-Android documentation
   - Check for existing community recipes
   - Create recipes based on existing examples

## Workflow Improvements

- Better error detection and reporting
- Detailed build logs
- Version verification
- System dependency checks
- Timeout increased to 150 minutes

