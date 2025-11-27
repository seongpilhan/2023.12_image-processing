# Frequency-Domain Feature Engineering for Enhanced COVID-19 and Pneumonia Classification

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

주파수 분석 기반 특징 추출을 통한 흉부 X-ray 이미지의 COVID-19/폐렴 분류 성능 향상 연구

## 🎯 Overview

이 프로젝트는 주파수 도메인 변환 기법을 활용하여 COVID-19와 폐렴 환자의 흉부 X-ray 이미지를 효과적으로 분류하는 딥러닝 모델을 제안합니다.

- **Dataset**: KAGGLE COVID-19 Radiography Database
  - COVID-19: 1,345 images
  - Viral Pneumonia: 1,345 images
- **Model**: Inception ResNet v2
- **Performance**: Raw Image 대비 1.16% accuracy 향상 (98.26% → 99.42%)

## 📄 Research Background

### Problem Statement

COVID-19와 폐렴 환자의 흉부 X-ray 이미지는 육안으로 구별하기 어렵습니다. 기존의 Raw Image 기반 딥러닝 분류 방법은 미세한 특징을 충분히 포착하지 못하는 한계가 있었습니다.

### Our Solution

주파수 도메인 변환을 통한 전처리로 이미지의 숨겨진 특징을 강조하여 분류 성능을 향상시킵니다.

## 🔬 Method

### Preprocessing Pipeline
```
Raw X-ray Image
    ↓
Lung Region Cropping (Masking)
    ↓
Frequency Domain Transform
    ├─ FFT (Fast Fourier Transform)
    ├─ PSD (Power Spectral Density)
    ├─ DCT (Discrete Cosine Transform)
    └─ DWT (Discrete Wavelet Transform)
    ↓
Classification (Inception ResNet v2)
```

### Image Processing Functions

#### 1. **FFT (Fast Fourier Transform)**
- 이미지를 주파수 도메인으로 변환
- 공간 도메인에서 보이지 않는 주파수 패턴 추출

#### 2. **PSD (Power Spectral Density)**
- 신호의 주파수 성분별 전력 분포 시각화
- FFT 결과의 magnitude를 평균 제곱으로 표현

#### 3. **DCT (Discrete Cosine Transform)**
- 실수 기반 변환으로 압축 효율 우수
- JPEG 압축 알고리즘의 핵심 기술

#### 4. **DWT (Discrete Wavelet Transform)**
- 다해상도 분석 가능
- 4개의 세부 계수로 분해 (Approximation, Horizontal, Vertical, Diagonal)
- **최고 성능 달성**: 99.42% accuracy

### Key Contributions

1. **주파수 도메인 기반 전처리**
   - 4가지 주파수 변환 함수의 체계적 벤치마킹
   
2. **폐 영역 추출 (Lung Cropping)**
   - 관심 영역만 선택하여 노이즈 제거
   
3. **성능 검증**
   - 차원 축소 알고리즘(t-SNE, UMAP, Wasserstein Distance)을 통한 분리도 분석

## 📊 Results

### Classification Performance

| Method | Test Accuracy | Improvement |
|--------|---------------|-------------|
| **Raw Image** | 98.26% | Baseline |
| **FFT** | 99.13% | +0.87% |
| **PSD** | 91.88% | -6.38% |
| **DCT** | 85.80% | -12.46% |
| **DWT** | **99.42%** | **+1.16%** |

### Dimensionality Reduction Analysis

**t-SNE Visualization**
- Raw Image와 전처리된 이미지들의 클러스터링 패턴 비교
- DWT가 가장 명확한 클래스 분리 보임

**UMAP Visualization**
- 고차원 데이터의 토폴로지 보존
- 클래스 간 경계 명확성 검증

**Wasserstein Distance**
- COVID-19 vs Pneumonia 분포 간 거리 측정
- DWT: 215,553.98 (가장 큰 분리도)

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. Download KAGGLE COVID-19 Radiography Database
2. Extract to `data/raw/`
```bash
data/
├── raw/
│   ├── COVID/
│   └── Viral Pneumonia/
```

### Training
```python
# Train with DWT preprocessing
python src/training/train_dwt.py --epochs 10 --batch_size 32

# Train with other transforms
python src/training/train.py --transform [fft|psd|dct|dwt]
```

### Inference
```python
from src.inference import classify_xray

result = classify_xray('path/to/xray.png', transform='dwt')
print(f"Prediction: {result['class']}, Confidence: {result['confidence']}")
```

## 📁 Repository Structure
```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Raw X-ray images
│   ├── processed/              # Preprocessed images
│   └── masks/                  # Lung segmentation masks
├── src/
│   ├── preprocessing/
│   │   ├── cropping.py        # Lung region extraction
│   │   ├── fft_transform.py   # FFT preprocessing
│   │   ├── psd_transform.py   # PSD preprocessing
│   │   ├── dct_transform.py   # DCT preprocessing
│   │   └── dwt_transform.py   # DWT preprocessing
│   ├── models/
│   │   └── inception_resnet_v2.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_dwt.py
│   ├── analysis/
│   │   ├── tsne_visualization.py
│   │   ├── umap_visualization.py
│   │   └── wasserstein_distance.py
│   └── inference.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_comparison.ipynb
│   └── 03_model_evaluation.ipynb
├── results/
│   ├── models/                # Trained model weights
│   └── figures/               # Visualization outputs
└── docs/
    └── presentation.pdf       # Research presentation
```

## ⚙️ Training Configuration

**Hyperparameters**
- Train samples: 1,000 per class
- Test samples: 345 per class
- Batch size: 32
- Epochs: 10
- Loss threshold: < 0.01
- Optimizer: Adam
- Learning rate: 1e-4

**Model Architecture**
- Base: Inception ResNet v2
- Input: 299×299×3
- Output: 2 classes (COVID-19, Pneumonia)

## 📈 Visualization Results

### t-SNE Clustering
전처리 방법별 특징 공간의 2D 투영 비교

### UMAP Projection
고차원 데이터의 토폴로지 구조 보존 시각화

### Wasserstein Distance
클래스 간 분포 차이의 정량적 측정

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow 2.x, Keras
- **Image Processing**: OpenCV, PIL, scikit-image
- **Frequency Analysis**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dimensionality Reduction**: scikit-learn, UMAP
- **Computing**: Google Colab (T4 GPU)

## 📚 References

1. [BMC Pulmonary Medicine - COVID-19 Chest X-ray](https://bmcpulmmed.biomedcentral.com/articles/10.1186/s12890-020-01286-5)
2. [KAGGLE COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/)
3. [Pneumonia Classification Using Deep Learning](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-021-00755-1)
4. [Inception-ResNet-v2 Paper](https://arxiv.org/abs/1602.07261v2)

## 👥 Authors

- **한성필** - AI Researcher, Korea Testing Laboratory
- **윤재영** - Co-researcher

## 📧 Contact

- Email: [your-email@example.com]
- Project Link: [https://github.com/username/covid-pneumonia-classification](https://github.com/username/repo)

## 📝 Citation
```bibtex
@inproceedings{han2024frequency,
  title={Frequency-Domain Feature Engineering for Enhanced COVID-19 and Pneumonia Classification},
  author={Han, Seongpil and Yoon, Jaeyoung},
  booktitle={Computer Vision Project},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- KAGGLE COVID-19 Radiography Database for providing the dataset
- Google Colab for computational resources
- TensorFlow team for the Inception ResNet v2 implementation

---

⭐ **Key Findings**: DWT 전처리를 통해 COVID-19와 폐렴 분류 정확도를 99.42%까지 향상시킴으로써, 주파수 도메인 분석이 의료영상 분류에 효과적임을 입증하였습니다.
