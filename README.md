# 🧠 Deep Learning Beginner

<div align="center">

![Deep Learning Banner](https://img.shields.io/badge/Deep%20Learning-Beginner's%20Guide-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

### *Your Journey into Deep Learning Starts Here*

**A comprehensive collection of hands-on notebooks designed to guide beginners through fundamental deep learning concepts and practical implementations.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

[Overview](#-overview) • [Learning Path](#-learning-path) • [Quick Start](#-quick-start) • [Models](#-model-implementations) • [Prerequisites](#-prerequisites)

</div>

---

## 📋 Overview

Reason behind **Deep Learning Beginner** - the reason I have developed this project is that, I used to have a lot of headache how I should deep-dive into the deep learning field. I was not even sure that, what was the very first step for it. So, now, I saved my first approach on entering deep learning. 

Why **Numpy** -  First of all ===>  The fundamental building block of the deep learning is numpy, we can say how neural networks are constructed and mathematically processed are directly related with numpy theories.

What are **Models & Data** -  Secondly ===>  You must choose the Tensorflow ( OR ) Pytorch. If your choice is Tensorflow, you are more aligned with this repo. I have built image-classification on labels ( technically, what is inside the image) by using the built-in datasets of datasets provided via TensorFlow’s Keras API.

### 🎯 What You'll Learn

- **NumPy Fundamentals**: Master array operations, broadcasting, and mathematical functions essential for deep learning
- **Neural Network Basics**: Understand the building blocks of neural networks through practical examples
- **Image Classification**: Build very basic deep learning models by using Tensorflow built-in image datasets
- **TensorFlow/Keras**: Get comfortable with the leading deep learning framework
- **Model Training Pipeline**: Learn data preprocessing, model building, training, and evaluation

---

## 🗺️ Learning Path
Apply your knowledge by building real image classification models.

| Model | Dataset | Task | Complexity |
|-------|---------|------|------------|
| **MNIST** | Handwritten Digits | 10-class classification | Data_Complexity-> ⭐ |
| **Fashion-MNIST** | Clothing Items | 10-class classification | Data_Complexity-> ⭐⭐ |
| **CIFAR-10** | Natural Images | 10-class classification | Data_Complexity-> ⭐⭐⭐ |

---

## 🚀 Quick Start

### Google Colab (Recommended)

1️⃣ **Clone or Download this Repository**
```bash
git clone https://github.com/thonedra-dev/deeplearning_beginner.git
```

2️⃣ **Upload to Google Drive** (Optional)
- Upload the repository folder to your Google Drive for easy access

3️⃣ **Open in Google Colab**
- Navigate to [Google Colab](https://colab.research.google.com/)
- Select `File` → `Upload notebook` or `Open notebook` → `GitHub`
- Upload individual `.py` or `.ipynb` files



---

## 📚 Model Implementations

### 🔢 MNIST - Handwritten Digit Recognition
**File:** `mnist_model.py`

The classic starting point for image classification. Learn to build a neural network that recognizes handwritten digits (0-9).

**What You'll Build:**
- Simple feedforward neural network
- Convolutional Neural Network (CNN)
- Model evaluation and visualization

**Dataset:** 60,000 training images + 10,000 test images (28×28 grayscale)

---

### 👕 Fashion-MNIST - Clothing Classification
**File:** `fashion_mnist.py`

Step up the challenge with more complex grayscale images of clothing items.

**What You'll Build:**
- Enhanced CNN architecture
- Data augmentation techniques
- Performance optimization strategies

**Dataset:** 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

### 🖼️ CIFAR-10 - Natural Image Classification
**File:** `ciphar10_model.py`

Master color image classification with real-world complexity.

**What You'll Build:**
- Deep CNN with multiple layers
- Batch normalization and dropout
- Advanced training techniques

**Dataset:** 60,000 color images (32×32 RGB) across 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 🛠️ Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of functions, loops, and data structures

### Software Requirements
- Python 3.9 or higher
- Google Colab account (free) **OR**
- Jupyter Notebook installed locally

### Key Libraries
```txt
numpy>=1.21.0
tensorflow>=2.10.0
matplotlib>=3.5.0
```

> 💡 **Pro Tip:** Google Colab comes with all these libraries pre-installed!

---

## 📁 Repository Structure

```
deeplearning_beginner/
├── numpy_practice/
│   ├── day_one_one.py
│   ├── day_one_two.py
│   ├── day_one_three.py
│   ├── day_one_four.py
│   ├── day_one_five.py
│   ├── day_two_one.py
│   ├── day_two_two.py
│   ├── day_two_three.py
│   ├── day_two_four.py
│   ├── day_three_one.py
│   ├── day_three_two.py
│   └── day_three_three.py
│
├── model_notebooks/
│   ├── mnist_model.ipynb
│   ├── fashion_mnist.ipynb
│   └── cifar10_model.ipynb
│
├── requirements.txt
└── README.md

```

---


## 📫 Contact & Support

<div align="center">

### **Thonedra**

[![Email](https://img.shields.io/badge/Email-thonedra.dev%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thonedra.dev@gmail.com)

*Passionate about making deep learning accessible to everyone*

</div>

### Need Help?

- 📧 Email me with questions: thonedra.dev@gmail.com
- 🐛 Report issues in the [Issues](https://github.com/thonedra-dev/deeplearning_beginner/issues) section
- 💬 Start a discussion in [Discussions](https://github.com/thonedra-dev/deeplearning_beginner/discussions)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2025 Thonedra
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🌟 Acknowledgments

- **TensorFlow Team** - For the incredible deep learning framework
- **NumPy Community** - For the foundation of scientific computing
- **Google Colab** - For providing free GPU access
- **Deep Learning Community** - For continuous inspiration and support

---

## 📊 Progress Tracking

Use this checklist to track your learning journey:

- [ ] Completed all Day 1 notebooks (NumPy basics)
- [ ] Completed all Day 2 notebooks (Array operations)
- [ ] Completed all Day 3 notebooks (Advanced concepts)
- [ ] Built and trained MNIST model
- [ ] Built and trained Fashion-MNIST model
- [ ] Built and trained CIFAR-10 model
- [ ] Achieved >95% accuracy on MNIST
- [ ] Achieved >85% accuracy on Fashion-MNIST
- [ ] Achieved >70% accuracy on CIFAR-10

---

<div align="center">

### 🎯 Ready to Start Your Deep Learning Journey? 🎯

**Clone this repo and begin with `day_one_one.py`!**

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)

**Built with passion for aspiring AI engineers | © 2025 Thonedra**

</div>