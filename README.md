# 24-D Local Von Neumann Entropy Local Feature Descriptor for HPatches Benchmark

This repository contains the standalone implementation of the **Quantum-Inspired 24-Dimensional Local Von Neumann Entropy Feature Descriptor** and its evaluation pipeline on the **HPatches (Homography Patches)** benchmark dataset.

---

## Architecture & Formulation

Given an input image and detected salient keypoints:

1. **Neighborhood Probing ($5 \times 5$ window):**  
   Around each keypoint $KP(x, y)$, 24 surrounding probe positions $P_1, P_2, \dots, P_{24}$ are defined in a fixed, deterministic layout (the center keypoint itself is excluded).
2. **Local Quantum Density Matrix ($3 \times 3$ window):**  
   Each probe position $P_i$ acts as the center of its own $3 \times 3$ patch ($9$ pixels). A normalized quantum density matrix $\rho_i \in \mathbb{C}^{9 \times 9}$ is constructed:
   $$\mathbf{q} = \frac{\mathbf{v} + \epsilon}{\sum (\mathbf{v} + \epsilon)}, \quad \rho_i = \text{diag}(\mathbf{q})$$
   which is Hermitian, positive semi-definite, and satisfies $\text{Tr}(\rho_i) = 1$.
3. **Von Neumann Entropy:**  
   The entropy of state $\rho_i$ is computed in bits:
   $$S(\rho_i) = -\text{Tr}(\rho_i \log_2 \rho_i) = -\sum_{k=1}^9 \lambda_k \log_2(\lambda_k)$$
4. **24-D Descriptor Vector:**  
   The final descriptor for each keypoint is the 24-element vector of probe entropies:
   $$F(KP) = \begin{bmatrix} S_1, S_2, \dots, S_{24} \end{bmatrix} \in \mathbb{R}^{24}$$
5. **Homography Estimation:**  
   Descriptors between reference and target images are matched via **Mutual Nearest Neighbor (MNN)** or Ratio Test. Homography matrices $\hat{\mathbf{H}}$ are estimated using **RANSAC**, and corner reprojection error is evaluated against ground-truth homographies $\mathbf{H}_{gt}$.

---

## Repository Structure

```
├── requirements.txt             # Lightweight dependencies (NumPy, SciPy, OpenCV, Matplotlib, TQDM)
├── README.md                    # This guide
├── .gitignore
├── src/
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── sobel_masks.py       # 5x5 8-direction Sobel directional masks
│   │   ├── sobel_operator.py    # 2D directional convolution
│   │   └── gradient.py          # Maximum gradient magnitude & dominant direction
│   └── feature/
│       ├── __init__.py
│       ├── density_matrix.py    # Intensity density matrix formulation
│       ├── von_neumann.py       # Von Neumann entropy calculation
│       ├── directional_variation.py # 8-direction directional variation & keypoint scoring
│       ├── keypoint_detection.py # Keypoint dataclass and NMS filtering
│       ├── entropy_descriptor.py # Core 24-D and 9-D descriptor extraction & matching
│       └── hpatches_adapter.py  # HPatches image I/O, H scaling, RANSAC, corner error, AUC
└── scripts/
    ├── hpatches_eval.py         # Main HPatches evaluation script
    └── hpatches_baselines.py    # Classical SIFT and ORB comparison baselines
```

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yaswanthvuppala/24-D-VonNeumon-Similarity-Q.git
cd 24-D-VonNeumon-Similarity-Q
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## HPatches Dataset Setup

Download the official HPatches sequences release (~4.2 GB):

### Linux / macOS / WSL:
```bash
mkdir -p data
cd data
wget -O hpatches-sequences-release.zip https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
unzip -q hpatches-sequences-release.zip
cd ..
```

### Windows (PowerShell):
```powershell
mkdir data
cd data
Invoke-WebRequest -Uri "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip" -OutFile "hpatches-sequences-release.zip"
Expand-Archive -Path "hpatches-sequences-release.zip" -DestinationPath "."
cd ..
```

### Or Clone Directly via Git:
```bash
git clone https://github.com/hpatches/hpatches-sequences.git data/hpatches-sequences-release
```

Expected directory hierarchy:
```
data/hpatches-sequences-release/
├── i_ajuntament/       # Illumination change sequence
│   ├── 1.ppm           # Reference image
│   ├── 2.ppm ... 6.ppm # Target images
│   ├── H_1_2 ... H_1_6 # 3x3 ground-truth homographies
├── v_adam/             # Viewpoint change sequence
└── ...                 # 116 sequences total (57 illumination, 59 viewpoint)
```

---

## Running the Benchmark

### 1. Evaluate QSED 24-D Descriptor
Run the complete evaluation across all 116 sequences (580 image pairs):
```bash
python scripts/hpatches_eval.py --dataset-dir data/hpatches-sequences-release
```

#### Advanced Options:
```bash
# Evaluate with L2 normalization (recommended)
python scripts/hpatches_eval.py --dimension 24 --norm l2 --matching mutual_nn

# Evaluate with unit-entropy normalization in [0, 1]
python scripts/hpatches_eval.py --dimension 24 --norm unit_entropy

# Evaluate the 9-D compact sliding-window descriptor
python scripts/hpatches_eval.py --dimension 9 --norm l2

# Generalized probe descriptor: 5x5 probes, each using a 3x3 entropy patch (24-D)
python scripts/hpatches_eval.py --n 5 --y 3 --norm l2

# Larger 7x7 probe neighborhood with 3x3 entropy patches (48-D)
python scripts/hpatches_eval.py --n 7 --y 3 --norm l2

# Custom keypoint budget and image resolution
python scripts/hpatches_eval.py --max-keypoints 512 --max-side 480 --ransac-thresh 3.0
```

### 2. Run Baselines (SIFT & ORB)
Compare with classical 128-D SIFT and 256-bit ORB descriptors:
```bash
python scripts/hpatches_baselines.py --dataset-dir data/hpatches-sequences-release
```

### 3. Evaluate on Oxford Affine Benchmark (~50 MB, 8 Sequences)
Auto-downloads the classic Mikolajczyk dataset from Oxford University and tests across 5 challenge types (Viewpoint, Scale/Rotation, Blur, Lighting, JPEG compression):
```bash
python scripts/oxford_eval.py
```

---

## Benchmark Comparison: RIPE Paper (Table 2 - HPatches)

The benchmark produces direct comparisons matching **Table 2** of the **RIPE** paper (*Künzel et al., arXiv:2507.04839*):

| Category | Method | Supervision | Dim | AUC@1px | AUC@3px | AUC@5px |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| (Semi-)Dense | Eff. LoFTR (CVPR'24) | Depth | Dense | 42.38% | 63.26% | 72.75% |
| | Mast3r (ECCV'24) | Depth | Dense | 29.63% | 54.04% | 66.86% |
| | RoMa (CVPR'24) | Depth | Dense | 38.38% | 63.75% | 73.37% |
| Sparse | SIFT (IJCV'04) | Handcrafted | 128 | 29.98% | 53.93% | 65.01% |
| | SuperPoint (CVPRW'18) | Homography Aug. | 256 | 36.61% | 56.92% | 67.28% |
| | DISK (NeurIPS'20) | Pose/Depth | 128 | 37.54% | 57.68% | 68.08% |
| | ALIKED (TIM'23) | Pose + Homography | 128 | 31.95% | 57.65% | 69.35% |
| | SiLK (ICCV'23) | Homography Aug. | 128 | 39.05% | 57.96% | 67.64% |
| | DeDoDe-B (3DV'24) | Pose | 256 | 39.88% | 60.59% | 70.36% |
| | **RIPE (Künzel et al. 2025)** | **Binary Image Pairs** | **256** | **37.93%** | **58.93%** | **69.20%** |
| **Proposed** | **QSED 24-D (Ours)** | **Local Density Matrix** | **24** | *Evaluated* | *Evaluated* | *Evaluated* |

---

## Output Metrics & Reports

The evaluation outputs:
- **Homography Estimation AUC:** Evaluated at 1px, 3px, and 5px thresholds via trapezoidal integration of the cumulative accuracy curve:
  $$\text{AUC}@T = \frac{1}{T} \int_0^T \text{Recall}(t) \, dt \times 100\%$$
- **Corner Reprojection Error:**
  $$\epsilon = \frac{1}{4} \sum_{i=1}^{4} \|\mathbf{H}_{gt}\mathbf{c}_i - \hat{\mathbf{H}}\mathbf{c}_i\|_2$$
- **Breakdown:** Separate performance for **Illumination (`i_*`)** vs **Viewpoint (`v_*`)** variations.
- **Reports Generated:**
  - `outputs/hpatches/hpatches_results.txt`: Formatted benchmark report with RIPE Table 2 comparison
  - `outputs/hpatches/hpatches_per_pair.csv`: Per-pair metrics (keypoint counts, matches, corner error)
  - `outputs/oxford/oxford_results.txt`: Oxford Affine benchmark report
