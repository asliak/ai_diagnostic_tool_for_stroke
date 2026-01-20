# AI-Enhanced Diagnostic Tool for the Stroke Teams

**Senior Design Project - Electronic and Communication Engineering**

## 🎯 The Problem:
Stroke is the second leading cause of death worldwide. In emergency scenarios, every minute counts—approximately 1.9 million neurons are lost per minute during a stroke.

During our research and interviews with medical professionals, we identified a critical bottleneck: **The "WhatsApp Paradox".**
In many hospitals, to get a quick second opinion, doctors film CT scan screens with their phones and send videos via WhatsApp. This leads to:
* Severe loss of image quality (resolution & contrast).
* Inability to adjust Hounsfield Units (windowing).
* Security and privacy vulnerabilities.

## 💡 Our Solution
We developed **Cerebrum Scanner** to replace this inefficient workflow. It is an AI-powered inference engine that takes raw medical data (DICOM/NIfTI), processes it correctly, and detects Intracranial Hemorrhage (ICH) with high precision.

Instead of compressed videos, we analyze the actual volumetric data.

## ⚙️ Technical Approach

### 1. Data Preprocessing (The Critical Step)
Medical images are not standard RGB photos. To make them "visible" to our model, we applied radiological windowing techniques:
* **Brain Window:** (Level: 40, Width: 80)
* **Subdural Window:** (Level: 80, Width: 200)
* **Slicing:** Converted 3D NIfTI volumes into 2D axial slices for efficient processing.

### 2. The Model (ResNet-50)
We utilized **Transfer Learning** with a **ResNet-50** architecture (pretrained on ImageNet).
* **Why ResNet-50?** We validated this choice against other models (during our preliminary Hymenoptera tests) and found it offered the best balance between speed and feature extraction depth.
* **Multi-Label Classification:** The model is modified to detect 6 different conditions simultaneously:
    * Epidural
    * Intraparenchymal
    * Intraventricular
    * Subarachnoid
    * Subdural
    * Any (General Hemorrhage presence)

## 📂 Repository Structure

```text
CerebrumScanner/
├── classification_module/   # The main logic. Contains 'classify_hemorage.py'
├── model_path/              # Weights of the trained ResNet-50 model
├── inputs/                  # Sample raw data (NIfTI/DICOM) for testing
├── outputs/                 # The AI generates JSON results and Heatmaps here
