# Master's thesis

## Title
Transfer learning for cloud semantic segmentation on satellite imagery

## Description
Cloud detection is a crucial step in the processing of optical satellite imagery. Deep learning models trained for cloud detection on data from one sensor often fail when applied to a different sensor due to differences in spectral and radiometric properties. Transfer learning techniques can help mitigate this problem. This work investigates transfer learning techniques for cloud semantic segmentation, focusing on the transfer from the Sentinel-2 imagery to the VENµS imagery. A model based on the U-Net architecture was trained on a subset of the CloudSEN12+ dataset and subsequently adapted to a cloud detection dataset for the VENµS satellite. Four transfer learning configurations were compared, ranging from transductive transfer to full network fine-tuning, along with four normalization strategies. In the transductive experiment, dynamic Z-score normalization achieved the best results, enabling cross-sensor generalization without any modification of the model weights. In the finetuning experiments, the best results were achieved using partial adaptation with frozen encoder weights, slightly surpassing the performance of a model trained from scratch on VENµS data. The results demonstrate that, when the source and target domains are sufficiently similar, a model adapted through transfer learning can match or even surpass the performance of a model trained exclusively on data from the target domain. These findings suggest that transfer learning can support the deployment of cloud detection models to new sensors while decreasing the need for large, sensor-specific annotated datasets.

## Student

Bc. Matěj Klimeš

## Supervisor

Ing. Ondřej Pešek, Ph.D.

## Readers
Prof. Arnon Karnieli, Ph.D.

## Defence
23.06.2026

## Text
[PDF](text/F1-DP-2026-Klimes-Matej-transfer_learning_compressed.pdf)

## Documentation

Documentation for preprocessing pipeline and evaluation scripts is available in [manual.md](docs/manual.md).

