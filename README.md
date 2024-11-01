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
|        Model         | Car | Person  | Rider  | Bus  | Truck | Train  | Bicycle  | Motor  | mAP  | Checkpoint  |
| :------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---------: |
|    best_model  | 61.9 | 45.2 | 46.1 | 47.1 | 25.0 | 49.7 | 36.8 | 29.9 | 42.7 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{det}$_1  | 61.662 | 44.779 | 46.336 | 48.460 | 24.686 | 48.984 | 35.809 | 28.912 | 42.4 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{det}$_2  | 61.589 | 45.092 | 46.221 | 48.266 | 24.518 | 48.972 | 36.098 | 29.403 | 42.5 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{MGFA}$_1  | 61.649 | 45.203 | 46.330 | 47.630 | 24.645 | 48.972 | 36.368 | 29.646 | 42.6 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{MGFA}$_2  | 61.567 | 45.114 | 46.097 | 48.171 | 24.438 | 48.972 | 36.112 | 29.9349 | 42.5 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{di}$_1  | 61.613 | 44.883 | 45.968 | 48.105 | 24.844 | 48.984 | 35.607 | 29.045 | 42.4 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{di}$_2  | 62.144 | 44.611 | 45.698 | 44.962 | 27.799 | 47.920 | 34.486 | 29.263 | 42.1 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{ds}$_1  | 62.161 | 44.620 | 45.822 | 44.263 | 27.909 | 47.626 | 34.405 | 28.983 | 42.0 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{ds}$_2  | 61.518 | 44.733 | 45.917 | 48.938 | 24.611 | 48.120 | 35.793 | 29.051 | 42.3 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{MI}$_1  | 61.641 | 44.922 | 46.126 | 48.360 | 24.781 | 48.972 | 36.167 | 29.137 | 42.5 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{MI}$_2  | 62.148 | 44.673 | 45.636 | 44.197 | 27.868 | 48.010 | 34.449 | 29.243 | 42.0 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{ILFD}$_1  | 61.583 | 44.845 | 46.134 | 48.600 | 24.727 | 49.283 | 35.953 | 29.001 | 42.5 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
|    ablation_model_hp_$L_{ILFD}$_2  | 61.636 | 44.997 | 46.249 | 47.704 | 24.499 | 48.473 | 35.770 | 28.684 | 42.3 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |


### YOLOX structure
The DRY continues to use the head structure of YOLOX for the detection head.
![image](./resources/YOLOX_structure.png)

### Code
The code for this program will be published soon.
