# 📱 How to Run on Your Phone - SIMPLE METHOD

## ✅ Easiest Way (Web Browser - Works Right Now!)

### Step 1: Install Flask (if not already installed)
```bash
pip install flask
```

### Step 2: Run the web server
```bash
./run_mobile.sh
```

Or manually:
```bash
python3 app_web.py
```

### Step 3: Find your computer's IP address
The script will show it, or run:
- **macOS/Linux**: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- **Windows**: `ipconfig` (look for IPv4 Address)

### Step 4: Open on your phone
1. Make sure your phone is on the **same WiFi network** as your computer
2. Open any web browser on your phone (Chrome, Safari, etc.)
3. Go to: `http://YOUR_IP:5000`
   - Example: `http://192.168.1.5:5000`
4. **Allow camera access** when prompted
5. Done! The app will work in your browser

### Features:
- ✅ Real-time concentration tracking
- ✅ Works on any phone with a browser
- ✅ No APK installation needed
- ✅ Same functionality as desktop version
- ✅ Mobile-optimized UI

---

## 🔧 Alternative: Build APK (Takes longer)

The GitHub Actions workflow will build an APK automatically. Check:
https://github.com/akshitmalik12/concentration-level-app/actions

If the build succeeds, download the APK from the Artifacts section and install it on your phone.

---

## 🆘 Troubleshooting

**Can't access from phone?**
- Make sure both devices are on the same WiFi
- Check your computer's firewall isn't blocking port 5000
- Try using your computer's IP address instead of localhost

**Camera not working?**
- Make sure you allowed camera access in the browser
- Try a different browser (Chrome usually works best)
- Check that no other app is using the camera

**Port 5000 already in use?**
- Change the port in `app_web.py` (line with `app.run(port=5000)`)
- Or kill the process: `lsof -ti:5000 | xargs kill -9`

