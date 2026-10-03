# Notebooks

Colab notebooks kept by the authors, published as they were (Colab user information removed). They are the baseline code referred to in the follow-up manuscript on the archived evaluation of the synthetic-data-trained Siamese network. Paths such as `/content/...` refer to the original Colab sessions, whose data are not included here.

| File | Content |
|---|---|
| `tez_Svm.ipynb` | SVM baseline: LIBSVM, three one-versus-rest RBF-kernel SVMs with probability estimates (C = 0.01, gamma = 1e-8) on the flattened raw pixel values of 105 x 105 images, without feature scaling. |
| `Untitled20.ipynb`, `Untitled26.ipynb` | Keras CNN baseline (AlexNet-style network, 224 x 224 x 3 input, three-class softmax, Adam, 50 epochs). See the manuscript, section "CNN and SVM baselines". Colab execution timestamps in the notebook metadata: 28 May 2021 and 14 Aug 2022 (`Untitled20`), 15 Nov 2021 (`Untitled26`). |
| `Untitled30.ipynb` | Keras LeNet-5-style notebook (SGD, batch size 32, 100 epochs, 28 x 28 input); contains no saved results. Executed 27 Nov 2021. |

The SNN code is in the repository root (`one_shot_*.py`). The notebooks do not contain the Dataset 1 or Dataset 2 images.
