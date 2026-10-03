# CNN output files

Output files of the Keras CNN baseline runs (see `../notebooks/`), one file per number of real labelled training images per class (the first line of each file gives that number, e.g. `100 Resim`). The test set of each run has 7,200 images (2,400 per class); the file reports test loss, test accuracy and, where present, per-class precision, recall, F1 and the confusion matrix.

| File | Test accuracy in the file |
|---|---|
| `cnn_output_2_images_per_class.txt` | 0.333 (a single class predicted for every test image) |
| `cnn_output_4_images_per_class.txt` | 0.333 (same) |
| `cnn_output_20_images_per_class.txt` | 0.333 (same) |
| `cnn_output_50_images_per_class.txt` | 0.584 |
| `cnn_output_100_images_per_class.txt` | 0.436 |
| `cnn_output_200_images_per_class.txt` | 0.584 |
| `cnn_output_400_images_per_class.txt` | 0.885 |
| `cnn_output_600_images_per_class.txt` | 0.845 |
| `cnn_output_1200_images_per_class.txt` | 0.930 |
| `cnn_output_2400_images_per_class.txt` | none (the file contains only the header and the test-set size) |

The files are published as kept by the authors. The training and test images (Dataset 1, Alidoost & Arefi, 2018) are not included.
