# SNN training logs

Training logs of the Siamese network (SNN) kept by the authors, published as they were. The first line of each file gives the number of real labelled training images per class (`<n> Resim OneShot`). A log line `[iteration] loss: ...` is printed every 10 iterations. Every 100 iterations the network is evaluated on 400 randomly drawn queries (one real query image and one real image per class, see the manuscript); the evaluation line gives `right`, `error` and `precision` of that evaluation, and the following line gives `TotalRight`, `TotalError` and `TotalPrecision`, the running total over all evaluations so far (every evaluation is counted, from the first checkpoint on). The learning rate is not printed in the logs.

| File | Images per class (first line) | Iterations | Evaluations | Precision of the last evaluation | Final `TotalPrecision` |
|---|---|---|---|---|---|
| `snn_log_600_images_per_class.txt` | 600 | 90,000 | 900 | 0.565 (highest: 0.643) | 0.533 |
| `snn_log_1200_images_per_class.txt` | 1200 | 90,000 | 900 | 0.810 (highest: 0.878) | 0.764 |
| `snn_log_10_images_per_class.txt` (found in the folder for 2,400 images per class) | 10 | 90,000 | 900 | 0.808 (highest: 0.823) | 0.350 |
| `snn_log_1_image_per_class_run1_200000_iterations.txt` | 1 | 200,000 | 2,000 | 0.135 (highest: 0.415) | 0.196 |
| `snn_log_1_image_per_class_run2_incomplete_43800_iterations.txt` | 1 | stops at 43,800 | 437 | 0.088 (highest: 0.458) | 0.149 (at 43,700) |

The logs for 200 and 400 images per class in the same storage are empty (0 bytes) and are not included. The 90,000-iteration setting with 400 queries per evaluation is that of the training scripts kept with these logs (learning rate 0.00006); which script produced the 1-image logs is not recorded.

These logs are not known to be the logs of the runs behind the accuracies tabulated in the manuscript (for example 0.850 at 600 and 0.864 at 1,200 images per class); the logs of those runs have not been located. See the manuscript, section "Limitations".
