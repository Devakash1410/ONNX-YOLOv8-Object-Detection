# Windows Setup Guide for YOLOv8 Object Detection

This guide provides step-by-step instructions for setting up the YOLOv8 Object Detection project on Windows.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Install Python](#step-1-install-python)
- [Step 2: Download the Project](#step-2-download-the-project)
- [Step 3: Create Virtual Environment](#step-3-create-virtual-environment)
- [Step 4: Install Dependencies](#step-4-install-dependencies)
- [Step 5: Download YOLOv8 Model](#step-5-download-yolov8-model)
- [Step 6: Run the Application](#step-6-run-the-application)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Windows 10 or Windows 11
- Administrator access (for Python installation)
- Internet connection
- At least 2GB of free disk space

---

## Step 1: Install Python

### Check if Python is Already Installed

Open **Command Prompt** or **PowerShell** and run:

```cmd
python --version
```

If Python 3.8 or higher is installed, you can skip to [Step 2](#step-2-download-the-project).

### Install Python (if needed)

1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check the box "Add Python to PATH" during installation
4. Click "Install Now"
5. After installation, verify by opening a new Command Prompt and running:
   ```cmd
   python --version
   ```

---

## Step 2: Download the Project

### Option A: Using Git (Recommended)

If you have Git installed:

**Command Prompt:**
```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection.git
cd ONNX-YOLOv8-Object-Detection
```

**PowerShell:**
```powershell
cd $HOME\Documents
git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection.git
cd ONNX-YOLOv8-Object-Detection
```

### Option B: Download ZIP

1. Go to [https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location (e.g., `Documents\ONNX-YOLOv8-Object-Detection`)
5. Open Command Prompt or PowerShell and navigate to the extracted folder:
   ```cmd
   cd path\to\ONNX-YOLOv8-Object-Detection
   ```

---

## Step 3: Create Virtual Environment

Creating a virtual environment keeps project dependencies isolated.

### Command Prompt:
```cmd
python -m venv venv
venv\Scripts\activate
```

### PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After activation, your prompt should show `(venv)` at the beginning.

---

## Step 4: Install Dependencies

With the virtual environment activated, install the required packages:

### For NVIDIA GPU Users (Recommended for better performance):
```cmd
pip install opencv-python imread-from-url onnxruntime-gpu cap-from-youtube streamlit
```

### For CPU-Only Users:
```cmd
pip install opencv-python imread-from-url onnxruntime cap-from-youtube streamlit
```

**Note:** The `requirements.txt` file includes `onnxruntime-gpu` by default. If you don't have an NVIDIA GPU, use `onnxruntime` instead.

### Verify Installation:
```cmd
pip list
```

You should see all the installed packages listed.

---

## Step 5: Download YOLOv8 Model

The YOLOv8m ONNX model file is required but not included in the repository due to its size (~99 MB).

### Option A: Convert from PyTorch Model (Recommended)

1. Install ultralytics package:
   ```cmd
   pip install ultralytics
   ```

2. Create a Python script to convert the model. Create a file named `convert_model.py`:
   ```python
   from ultralytics import YOLO
   
   # Download and convert YOLOv8m model
   model = YOLO("yolov8m.pt")
   model.export(format="onnx", imgsz=[480, 640])
   ```

3. Run the conversion script:
   ```cmd
   python convert_model.py
   ```

4. Move the generated `yolov8m.onnx` file to the `models` folder:
   ```cmd
   move yolov8m.onnx models\yolov8m.onnx
   ```

### Option B: Use Google Colab

If you prefer not to install additional packages locally:

1. Open the Colab notebook: [YOLOv8 ONNX Conversion](https://colab.research.google.com/drive/1-yZg6hFg27uCPSycRCRtyezHhq_VAHxQ?usp=sharing)
2. Follow the instructions in the notebook to convert the model
3. Download the generated `yolov8m.onnx` file
4. Place it in the `models` folder of your project

### Verify Model File

Check that the model file exists:

**Command Prompt:**
```cmd
dir models\yolov8m.onnx
```

**PowerShell:**
```powershell
Test-Path models\yolov8m.onnx
```

---

## Step 6: Run the Application

### Run the Streamlit Web App

With your virtual environment activated:

```cmd
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Run Individual Scripts

**Image Detection:**
```cmd
python image_object_detection.py
```

**Webcam Detection:**
```cmd
python webcam_object_detection.py
```

**Video Detection:**
```cmd
python video_object_detection.py
```

---

## Troubleshooting

### Issue: "python is not recognized as an internal or external command"

**Solution:** Python is not in your PATH. Either:
- Reinstall Python and check "Add Python to PATH"
- Or manually add Python to PATH:
  1. Search for "Environment Variables" in Windows
  2. Click "Environment Variables"
  3. Under "System variables", find "Path" and click "Edit"
  4. Add Python installation path (e.g., `C:\Users\YourName\AppData\Local\Programs\Python\Python311`)

### Issue: "Cannot activate virtual environment in PowerShell"

**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:** Make sure your virtual environment is activated and reinstall opencv-python:
```cmd
pip install --upgrade opencv-python
```

### Issue: "ONNX Runtime error" or slow performance

**Solution:** 
- If you have an NVIDIA GPU, ensure you have CUDA installed and use `onnxruntime-gpu`
- If you don't have a GPU or CUDA, use `onnxruntime` instead:
  ```cmd
  pip uninstall onnxruntime-gpu
  pip install onnxruntime
  ```

### Issue: "Model file not found"

**Solution:** Ensure the model file is in the correct location:
- The default path is `models/yolov8m.onnx`
- Check the file exists using: `dir models\yolov8m.onnx`
- If using a different model name or location, update the path in the Streamlit app sidebar

### Issue: Webcam not working

**Solution:**
- Ensure your webcam is not being used by another application
- Check Windows Privacy Settings:
  1. Go to Settings > Privacy > Camera
  2. Enable "Allow apps to access your camera"
- Try running the script as Administrator

### Issue: "ImportError: DLL load failed"

**Solution:** Install Microsoft Visual C++ Redistributable:
- Download from [Microsoft's website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Install both x64 and x86 versions

### Issue: Streamlit app won't start

**Solution:**
1. Check if port 8501 is already in use:
   ```cmd
   netstat -ano | findstr :8501
   ```
2. If in use, either close the other application or run Streamlit on a different port:
   ```cmd
   streamlit run app.py --server.port 8502
   ```

### Issue: Low FPS or slow detection

**Solution:**
- Use a smaller model (e.g., yolov8n.onnx or yolov8s.onnx)
- Reduce input image size
- If you have an NVIDIA GPU, ensure you're using `onnxruntime-gpu` with CUDA installed
- Close other resource-intensive applications

---

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [project's GitHub Issues](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/issues)
2. Review the main [README.md](README.md) for additional information
3. Ensure all dependencies are correctly installed: `pip list`
4. Verify Python version compatibility: `python --version` (should be 3.8+)

---

## Deactivating Virtual Environment

When you're done working on the project, deactivate the virtual environment:

```cmd
deactivate
```

To reactivate it later, navigate to the project directory and run:

**Command Prompt:**
```cmd
venv\Scripts\activate
```

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1