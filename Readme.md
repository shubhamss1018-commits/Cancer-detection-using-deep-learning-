1. Project structure
Backend/ – Python training + FastAPI API
frontend/ – React + Vite UI
2. Prerequisites
Python 3.14 (your current install)
Node.js + npm
NVIDIA GPU is optional (you’re currently using CPU)
3. Backend setup (one‑time)
Open PowerShell.
Go to backend:
cd "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection\Backend"
Create and activate virtualenv:
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
Install dependencies:
pip install -r requirements.txt
Install CPU PyTorch (if needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
4. Training the model
From the activated backend venv:

cd "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection\Backend"
# Example: 1 epoch (quick test)
python main.py --mode train --data_dir "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection" --num_epochs 1
# Example: more epochs, EfficientNet backbone
python main.py --mode train --data_dir "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection" --backbone efficientnet --num_epochs 10
Training creates a run folder like results\run_YYYYMMDD_HHMMSS\.
The best checkpoint is at results\run_...\checkpoints\best_model.pth.
5. Prepare model for API
After a training run finishes:

cd "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection\Backend"
mkdir models -Force
copy ".\results\run_YYYYMMDD_HHMMSS\checkpoints\best_model.pth" ".\models\best_model.pth" /Y
(Replace run_YYYYMMDD_HHMMSS with your latest run folder.)

The API automatically loads models\best_model.pth.

6. Run the backend API (FastAPI)
cd "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection\Backend"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python api.py
FastAPI listens on http://localhost:8000.
Keep this terminal window open.
7. Frontend setup & dev server
Open another PowerShell window.
Go to frontend:
cd "C:\Users\sidde\OneDrive\Desktop\cancer po2\Cancer-detection\frontend"
First time only, install deps:
npm install
Start dev server:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm run dev
Open the shown URL (usually http://localhost:5173) in your browser.
8. Using the web app
Ensure backend is running (Application startup complete. in its terminal).
Open the frontend URL.
Check System status → Backend API shows Online.
Steps:
Click the upload area and select a histopathology image (PNG/JPG).
Click Run prediction.
View:
Predicted stage + probabilities.
Predicted progression risk + probabilities.
Grad‑CAM heatmap overlay.
9. Running from VS Code
Open folder Cancer-detection in VS Code.

Use Terminal → New Terminal:

Terminal 1 (backend):

cd Backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python api.py
Terminal 2 (frontend):

cd frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm run dev
Open the frontend URL and use the app as above.

