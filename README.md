# 📦 Ingrowth-Segmentation-DSA-Plugin

**Ingrowth-Segmentation-DSA-Plugin** is a Python package for image segmentation of ingrown structures in microscopy images of Cerebral Aneurysms. It is designed for high-resolution scientific image analysis with support for large formats, deep learning models, and preprocessing pipelines.

> ⚠️ **Note:** This plugin is intended for deployment and execution within a [Digital Slide Archive (DSA)](https://digitalslidearchive.github.io/) environment.  
> It **cannot be run directly on a local machine** and is designed to be integrated with the DSA system using `girder-slicer-cli-web` and `ctk-cli`.

---

## 🚀 Features

- 🧠 Deep learning segmentation with [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- 🖼 Handles large microscopy images with `imageio`, `Pillow`, and `opencv-python`
- 📊 Rich support for data handling via `pandas`, `openpyxl`, and `xlrd`
- ⚙️ Seamless integration with `girder-client`, `girder-slicer-cli-web`, and `ctk-cli`
- 🧪 Includes preprocessing and augmentation with `albumentations`

---

## 🛠 Installation (for development or DSA deployment)

To install the plugin manually (for development or DSA configuration):

```bash
git clone https://github.com/SarderLab/Ingrowth-Segmentation-DSA-Plugin.git && \
cd Ingrowth-Segmentation-DSA-Plugin && \
pip install .
```

> **Note:** This project uses `setuptools_scm` for versioning. Ensure it is installed if building from source.

---

## 📁 Repository Structure

```
Ingrown/
├── __init__.py
├── ... (core segmentation modules)
tests/
setup.py
README.md
```

---

## 🧪 Requirements

Major dependencies (automatically installed):

- `numpy`, `pandas`, `scikit-image`, `scikit-learn`
- `opencv-python`, `Pillow`, `imageio`
- `segmentation-models-pytorch`, `albumentations`
- `girder-client`, `girder-slicer-cli-web`, `ctk-cli`
- `tqdm`, `openpyxl`, `xlrd`, `joblib`

Optional (commented in `setup.py`):

- `torch`, `torchvision`, `matplotlib`, `shapely`, `dask`, `tifffile`, etc.

---

## 💻 Usage

This package is designed to be executed as a plugin inside the Digital Slide Archive (DSA) using the slicer CLI interface.  
Local usage or direct script execution is **not supported**.

---

## 👤 Authors

**Sayat Mimar**  
📧 [sayat.mimar@ufl.edu](mailto:sayat.mimar@ufl.edu)  
🧪 Developed at [Computational Microscopy Imaging Laboratory](https://cmilab.nephrology.medicine.ufl.edu/), University of Florida

**Fatemeh Afsari**  
📧 [f.afsari@ufl.edu](mailto:f.afsari@ufl.edu)  
🧠 Developed at [Computational Microscopy Imaging Laboratory](https://cmilab.nephrology.medicine.ufl.edu/), University of Florida

---

## 📄 License

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

---

## 📌 Citation

If you use this code for research, please cite this repository or related publications (coming soon).
