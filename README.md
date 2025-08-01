# 📦 Ingrown-Segmentation

**Ingrown-Segmentation** is a Python package for image segmentation of ingrown structures in microscopy images of Cerebral Aneurysms. It is designed for high-resolution scientific image analysis with support for large formats, deep learning models, and preprocessing pipelines.

---

## 🚀 Features

- 🧠 Deep learning segmentation with [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- 🖼 Handles large microscopy images with `imageio`, `Pillow`, and `opencv-python`
- 📊 Rich support for data handling via `pandas`, `openpyxl`, and `xlrd`
- ⚙️ Ready for integration with `girder-client`, `girder-slicer-cli-web`, and `ctk-cli`
- 🧪 Includes preprocessing and augmentation with `albumentations`

---

## 🛠 Installation

Install via `pip`:

```bash
pip install git+https://github.com/SarderLab/Ingrown-Segmentation.git
```

Or clone the repository and install manually:

```bash
git clone https://github.com/SarderLab/Ingrown-Segmentation.git && \
cd Ingrown-Segmentation && \
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
README.rst
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

## 💻 Usage Example

```python
from Ingrown import your_module

# Example function call
your_module.run_segmentation("path/to/image.tif")
```

> More examples and CLI usage will be added soon.

---

## 👤 Authors

**Sayat Mimar**  
📧 [sayat.mimar@ufl.edu](mailto:sayat.mimar@ufl.edu)  
🧪 Developed at [Computational Microscopy Imaging Laboratory](https://cmilab.nephrology.medicine.ufl.edu/), University of Florida


- **Fatemeh Afsari**  
  📧 [f.afsari@ufl.edu](f.afsari@ufl.edu)  
  🧠 Developed at [Computational Microscopy Imaging Laboratory](https://cmilab.nephrology.medicine.ufl.edu/), University of Florida

---

## 📄 License

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

---

## 📌 Citation

If you use this code for research, please cite this repository or related publications (coming soon).
