# DRY：Coarse-to-Fine Domain Adaptation Object Detection with Feature Disentanglement

### Abstract
Domain adaptation object detection (DAOD) uses the labeled data of one scene (i.e., the source domain) and the unlabeled data of another unfamiliar scene (i.e., the target domain) to train the cross-domain object detector. Most existing methods align the overall distribution of features by adversarial adaptive methods. Despite their success, these methods are primarily designed for two-stage detectors that are challenging to deploy, resulting in limited practical applications. In addition, owing to the instability of adversarial domain discriminator training, inducing the detector is difficult using only an adversarial adaptive strategy to extract instance-level domain-invariant features to align the overall distribution. To address these issues, we propose a new cross-domain object detection framework based on the You Only Look Once (YOLO) series of algorithms named Disentanglement Representation YOLO (DRY). The developed method achieves feature disentanglement in the channel dimension and spatial dimensions through domain-invariant feature disentanglement (DIFD) and instance-level feature disentanglement (ILFD) modules, respectively, prompting the detector to extract domain-invariant features. Experiments demonstrate that our model outperforms existing methods. It achieved an average accuracy value of 42.7 on the Cityscapes to FoggyCityscapes benchmark and significantly outperformed all other methods on human and car objects. The average accuracy values of 49.0 and 49.5 achieved on the SIM10K to Cityscapes and KITTI to Cityscapes scenarios, respectively, are superior to those of existing methods. Extensive experimental results on various datasets verify that the proposed DRY method is effective and widely applicable.

![image](./resources/da-net.png)

### Contributions
1) The disentanglement representation YOLO (DRY) framework is proposed to implement domain adaptation object detection for a single-stage YOLO series detector.Compared with works based on two-stage detectors, cross-domain detection based on an efficient YOLO series detector has higher inference speed and efficiency.
2) The DIFD module is proposed for inducing backbone extraction of domain-invariant features by disentangling domain-specific and domain-invariant parts from backbone features.
3) The ILFD module is proposed for guiding the adversarial domain discriminator to concentrate on instance-level domain-invariant features by disentangling the foreground and background parts.
4) Extensive experiments performed in common cross-domain scenarios qualitatively and quantitatively demonstrate the effectiveness of DRY.

### Model
Due to the inability to upload large files in Guthub, we have uploaded our optimal model to Google Cloud Drive at the [link](https://drive.google.com/file/d/1iqEhEscleOlL_tr6G2JpmjknY9Ik4zt5/view?usp=sharing)

The link to the optimal model for other fog densities is [link](https://drive.google.com/drive/folders/1DPevWOIVhNHuvoJy5j_UpqJSMFFE0Ca6?usp=sharing)

### YOLOX structure
The DRY continues to use the head structure of YOLOX for the detection head.
![image](./resources/YOLOX_structure.png)

### Comparison with other comparative methods under different fog density conditions
Due to the fact that some of the compared methods do not disclose any source code, as well as the fact that some of the methods lack the necessary documentation in their repositories, we are actively contacting the authors of these methods. The comparison methods that we have successfully reproduced in full so far are listed below:

The form will be updated on an ongoing basis.

| Method | fog density coefficient   | Car | Person | Rider | Bus | Truck | Train | Bicycle | Motor | mAP |
|--------|---------------------------|-----|--------|-------|-----|-------|-------|---------|-------|-----|
| EPM    |         0.005             | 67.9|  48.5  |  49.0 | 55.7| 34.7  |  43.5 |   40.4  |  35.9 | 46.9|
| EPM    |         0.01              | 63.5|  47.2  |  46.1 | 50.2|  28.5 |  37.1 |   41.6  |  31.2 | 43.2|
| EPM    |         0.02              | 57.3|  39.9  |  38.1 | 50.7|  28.7 |  37.2 |   34.2  |  30.2 | 39.5|
|--------|---------------------------|-----|--------|-------|-----|-------|-------|---------|-------|-----|
| our    |         0.005             | 70.4|  51.7  |  45.0 | 55.3|  32.5 |  58.2 |   42.1  |  39.2 | 49.3|
| our    |         0.01              | 67.4|  49.7  |  47.8 | 49.6|  25.9 |  51.1 |   41.8  |  37.4 | 46.3|
| our    |         0.02              | 61.9|  45.2  |  46.1 | 47.1|  25.0 |  49.7 |   36.8  |  29.9 | 42.7|

### Code
The code for this program will be published soon.
