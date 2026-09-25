Write-Host "=== QGAN-LLM Environment Rebuild Script ===" -ForegroundColor Cyan

# -----------------------------
# 1. Install Python 3.11 (if missing)
# -----------------------------
Write-Host "`nChecking Python 3.11 installation..."

$python311 = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe"

if (-Not (Test-Path $python311)) {
    Write-Host "Python 3.11 not found. Installing via winget..." -ForegroundColor Yellow
    winget install -e --id Python.Python.3.11 --silent
    
    # Fallback path check if winget installed system-wide or added to PATH
    if (-Not (Test-Path $python311)) {$python311 = (Get-Command python3.11 -ErrorAction SilentlyContinue).Path
    }
} else {
    Write-Host "Python 3.11 already installed." -ForegroundColor Green
}

if (-Not $python311 -or -Not (Test-Path$python311)) {
    Write-Error "Python 3.11 executable could not be found. Please install Python 3.11 manually."
    exit 1
}

# -----------------------------
# 2. Remove old venv
# -----------------------------
Write-Host "`nRemoving old virtual environment..." -ForegroundColor Yellow

if (Test-Path ".\venv") {
    Remove-Item -Recurse -Force ".\venv"
    Write-Host "Old venv removed." -ForegroundColor Green
} else {
    Write-Host "No existing venv found." -ForegroundColor Green
}

# -----------------------------
# 3. Create new venv
# -----------------------------
Write-Host "`nCreating new virtual environment using Python 3.11..." -ForegroundColor Cyan

& $python311 -m venv venv
Write-Host "New venv created." -ForegroundColor Green

# -----------------------------
# 4. Upgrade pip (Fixes Windows file lock error)
# -----------------------------
Write-Host "`nUpgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# -----------------------------
# 5. Install core scientific stack (binary wheels only)
# -----------------------------
Write-Host "`nInstalling NumPy (binary wheel)..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install numpy==2.4.4 --only-binary=:all:

Write-Host "`nInstalling PyTorch..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nInstalling scientific packages..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install pandas scipy scikit-learn statsmodels pyyaml requests

Write-Host "`nInstalling Pennylane + Lightning..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install pennylane pennylane-lightning

# -----------------------------
# 6. Install project requirements
# -----------------------------
Write-Host "`nInstalling project requirements.txt..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install -r requirements.txt

# -----------------------------
# 7. Verify environment
# -----------------------------
Write-Host "`nRunning environment verification..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -c "import dukascopy; print('Dukascopy module loaded successfully from:', dukascopy.__file__)"
.\venv\Scripts\python.exe -m scripts.verify_environment

Write-Host "`n=== Environment rebuild complete ===" -ForegroundColor Green