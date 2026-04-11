
```markdown
# Sign Language AI System  
**Real-time Sign Language Recognition + Text-to-Sign Animation + Video Recording**  
**100% Working on Windows 10 / 11 (2025)**  

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
| **MySQL Server**    | 8.0 or 8.4        | Community edition                 |

---

### Step-by-Step Setup (Windows)

#### 1. Install Python 3.11 (64-bit)
Download: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
**Check "Add Python to PATH"** during install!

#### 2. Install XAMPP SERVER (APACHE & MySQL Server)
Download:[Download](https://www.apachefriends.org/)
Install **XAMPP**
Start MySQL & APACHE Server
GOTO PHPMYADMIN and create database: "sign_ai"

#### 3. Download the Project
```bash
git clone [GITHUB](https://github.com/ashab20/sign-language-ai-translator)
cd sign_ai
```
or download as ZIP → extract

#### 4. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv\Scripts\activate
```
You’ll see `(venv)` in your terminal

#### 5. Install All Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `ttkbootstrap` fails:
```bash
pip install ttkbootstrap==1.18.1
```

> **Tkinter is already included** with Python on Windows – no extra install needed

#### 6. Create `.env` file (in project root)
```env
DB_USER=root
DB_PASSWORD=your_mysql_password_here
DB_HOST=localhost
DB_PORT=3306
DB_NAME=sign_ai
```

#### 7. Run the App!
```bash
python main.py
```

First run automatically:
- Creates database `sign_ai`
- Creates table `gestures`
- Asks for camera permission (click **Allow**)

---

### Quick Usage Guide

| Button             | What it does                                      |
|--------------------|----------------------------------------------------|
| Record Gesture     | Type sign name → hold hand → saves 100 frames     |
| Train Model        | Trains the AI (need ≥10 samples per sign)         |
| Use AI             | Real-time live recognition                        |
| Play Live          | Animate typed text (e.g., "HELLO WORLD")          |
| Record Video       | Saves the animation as MP4 in `recordings/`       |
| Play Last          | Replays the last recorded video                   |

---

### Project Folders After Running

```
sign_ai/
├── data/
│   └── model.pkl              ← your trained model
├── recordings/
│   └── HELLO_WORLD_20251120_221300.mp4   ← saved videos
├── .env                       ← database password
└── venv/                      ← virtual environment
```

---

### Troubleshooting (Windows)

| Problem                              | Fix                                           |
|--------------------------------------|------------------------------------------------|
| Access denied (MySQL)                | Check password in `.env`                       |
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
