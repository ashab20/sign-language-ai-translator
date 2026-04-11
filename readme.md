# Sign Language AI System  
**Real-time Sign Language Recognition + Text-to-Sign Animation + Video Recording**  
**SQLite + offline text-to-speech — Windows / macOS**  

```
   _____ _                   __                                  
  / ____(_)                 / /                                  
 | (___  _  __ _ _ __   __ | | ___   __ _ _   _ _ __   __ _ _   _ 
  \___ \| |/ _` | '_ \ / _` | |/ _ \ / _` | | | | '_ \ / _` | | | |
  ____) | | (_| | | | | (_| | | (_) | (_| | |_| | | | | (_| | |_| |
 |_____/|_|\__, |_| |_|\__,_|_|\___/ \__, |\__,_|_| |_|\__,_|\__, |
            __/ |                     __/ |                   __/ |
           |___/                     |___/                   |___/ 
```

---

### Compatible Versions (Tested & Guaranteed)

| Component           | Version           | Notes                              |
|---------------------|-------------------|------------------------------------|
| **Python**          | 3.12.3 (64-bit)   | Must be Python 3.12               |
| **Windows**         | 10 / 11 (64-bit)  | Fully tested                      |
| **OpenCV**          | 4.10.0.84         | Pre-built wheels                  |
| **MediaPipe**       | 0.10.14           | Works on Windows + Apple Silicon  |
| **SQLite**          | (built-in file DB) | `data/sign_ai.db` — no server install |

---

### Step-by-Step Setup (Windows)

#### 1. Install Python 3.11 (64-bit)
Download: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
**Check "Add Python to PATH"** during install!

#### 2. Download the Project
```bash
git clone [GITHUB](https://github.com/ashab20/sign-language-ai-translator)
cd sign_ai
```
or download as ZIP → extract

#### 3. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv\Scripts\activate
```
You’ll see `(venv)` in your terminal

#### 4. Install All Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `ttkbootstrap` fails:
```bash
pip install ttkbootstrap==1.18.1
```

> **Tkinter is already included** with Python on Windows – no extra install needed

#### 5. Optional `.env` (in project root)

Default database path is `data/sign_ai.db`. To use another path:

```env
SQLITE_PATH=data/sign_ai.db
```

On **macOS**, text-to-speech uses the system `pyttsx3` backend (may require allowing microphone/speech in System Settings if prompted).

#### 6. Run the App!
```bash
python main.py
```

First run automatically:
- Creates SQLite file `data/sign_ai.db` (if missing)
- Creates table `gestures`
- Asks for camera permission (click **Allow**)

**Important:** Each sample stores **both hands** when visible (126 numbers: right then left in camera space; missing hand is zeros). One-handed signs still work. If you change feature layout or have old samples, **record again and run Train model** so `data/model.pkl` matches.

---

### Quick Usage Guide

| Button             | What it does                                      |
|--------------------|----------------------------------------------------|
| Record gesture     | Type sign name → hold hand → saves 100 frames     |
| Train model        | Trains the AI (need ≥10 samples per sign)         |
| Use AI             | Live recognition with smoothed labels               |
| Stop AI            | Stops the camera / prediction loop                  |
| Read aloud (toggle)| When a sign **ends** (hand leaves frame), speaks the last completed sign |
| Play live          | Animate typed text (e.g., "HELLO WORLD")          |
| Record video       | Saves the animation as MP4 in `recordings/`       |
| Play last          | Replays the last recorded video                   |

---

### Project Folders After Running

```
sign_ai/
├── data/
│   └── model.pkl              ← your trained model
├── recordings/
│   └── HELLO_WORLD_20251120_221300.mp4   ← saved videos
├── sign_ai.db                 ← inside `data/` (gesture samples)
└── venv/                      ← virtual environment
```

---

### Troubleshooting (Windows)

| Problem                              | Fix                                           |
|--------------------------------------|------------------------------------------------|
| SQLite “database is locked”          | Close other copies of the app using the same file |
| Table doesn't exist                  | Just run again – auto-created                  |
| Camera not working                   | Allow camera in Windows Privacy Settings       |
| Black screen                         | Click "Use AI" first to test camera            |
| `ttkbootstrap` error                 | Run: `pip install ttkbootstrap`                |

---

### Tested & Confirmed Working

| OS               | Python   | Status     | Date Tested     |
|------------------|----------|------------|-----------------|
| Windows 11 Pro   | 3.11.9   | Perfect    | Nov 20, 2025    |
| Windows 10 Home  | 3.11.7   | Perfect    | Nov 18, 2025    |

---

**Made with love in Bangladesh**  
**Ashab – November 2025**

**Star this repo if it helped you!**  

Happy signing!  
Let’s make communication accessible for everyone.
