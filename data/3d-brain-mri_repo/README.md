---
license: cc-by-4.0
---

![alt text](assets/hpe_logo.png "HPE Logo")


# PDK - Pachyderm | Determined | KServe
## 3D Brain MRI Example
**Date/Revision:** April 30, 2024

This dataset is based on the **UCSF-PDGM: The University of California San Francisco Preoperative Diffuse Glioma MRI** research dataset, which can be found here:
- https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

The original dataset contains data from 495 unique subjects. The dataset is formed by taking several MRI scans for each patient, “skull stripping” the scan (leaving just the brain image), and de-identifying the patient. The result is 4 MRI volumes per subject, as well as a target segmentation mask. In the [data](https://huggingface.co/datasets/determined-ai/3d-brain-mri/tree/main/data) folder, you will find a small subset of the data from 87 subjects, which can be used to train a segmentation model.

A sample payload for inference can be found in the [sample-payload](sample-payload) under [3d-brain.json](sample-payload/3d-brain.json). [Here are the full instructions for how to deploy this on HPE's end-to-end ML platform](https://github.com/determined-ai/pdk/tree/dev_3dmri/examples/3d-brain-mri)

![gif](assets/all_mri_mask.gif)