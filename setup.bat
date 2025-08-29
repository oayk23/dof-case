@echo off
echo ========================================
echo Dof Case Study Kurulum Scripti
echo ========================================

:: Hata durumunda scripti durdur
setlocal enabledelayedexpansion

echo.
echo [1/6] Virtual Environment 'dof' oluşturuluyor...
python -m venv dof
if errorlevel 1 (
    echo HATA: Virtual environment oluşturulamadı!
    pause
    exit /b 1
)
echo Virtual environment 'dof' başarıyla oluşturuldu.

echo.
echo [2/6] Virtual Environment aktive ediliyor...
call dof\Scripts\activate.bat
if errorlevel 1 (
    echo HATA: Virtual environment aktive edilemedi!
    pause
    exit /b 1
)
echo Virtual environment aktive edildi.

echo.
echo [3/6] Pip güncelleniyor...
python -m pip install --upgrade pip

echo.
echo [4/6] Requirements.txt kurulumu...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo HATA: Requirements.txt kurulumu başarısız!
        pause
        exit /b 1
    )
    echo Requirements.txt başarıyla kuruldu.
) else (
    echo UYARI: requirements.txt dosyası bulunamadı!
    echo Devam ediliyor...
)

echo.
echo [5/6] Ollama Qwen modeli indiriliyor...
ollama pull qwen3:1.7b
if errorlevel 1 (
    echo HATA: Ollama modeli indirilemedi! Ollama kurulu olduğundan emin olun.
    echo Devam ediliyor...
)

echo.
echo [6/6] Models klasörü oluşturuluyor...
if not exist models mkdir models
cd models

echo Hugging Face modellerini indiriliyor...
echo Bu işlem internet bağlantınıza bağlı olarak uzun sürebilir...

echo.
echo Model 1: all-MiniLM-L6-v2
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
if errorlevel 1 (
    echo HATA: all-MiniLM-L6-v2 modeli indirilemedi!
)

echo.
echo Model 2: faster-whisper-base
git clone https://huggingface.co/Systran/faster-whisper-base
if errorlevel 1 (
    echo HATA: faster-whisper-base modeli indirilemedi!
)

echo.
echo Model 3: xtts-v2
git clone https://huggingface.co/coqui/XTTS-v2
if errorlevel 1 (
    echo HATA: xtts-v2 modeli indirilemedi
)

cd ..

echo.
echo [7/7] PyTorch CUDA versiyonu kuruluyor...
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo HATA: PyTorch CUDA versiyonu kurulumu başarısız!
    pause
    exit /b 1
)

echo.
echo [8/8] Indexler hazırlanıyor...
python indexer.py

echo.
echo ========================================
echo KURULUM TAMAMLANDI!
echo ========================================
echo.
echo Virtual Environment: dof
echo Aktive etmek için: call dof\Scripts\activate.bat
echo Deaktive etmek için: deactivate
echo.
echo Kurulu bileşenler:
echo - Python Virtual Environment (dof)
echo - Requirements.txt paketleri
echo - Ollama Qwen3:1.7b modeli
echo - Hugging Face modelleri (models/ klasöründe)
echo - PyTorch CUDA versiyonu
echo.

pause