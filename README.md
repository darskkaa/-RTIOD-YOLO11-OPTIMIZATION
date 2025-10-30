# **RTIOD: Robust Thermal-Image Object Detection challenge**

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

not official!!!
<img src="assets/chal_logo.png" alt="drawing" width="50%">

## Information

Ready to stress-test your detector in the wild? The 6th Real World Surveillance Workshop at WACV 2026 [RWS@WACV2026](https://vap.aau.dk/rws/) introduces a new challenge on Robust Thermal-Image Object Detection using the LTDv2 dataset to advance multi-object detection performance under long-term thermal drift. Real-world thermal imaging is messy: sensors re-calibrate, ambient temperatures swing, and weather keeps changing, which slowly erode detector performance.

The objectives, rules, and challenge submission are handled on the Codabench challenge [page](https://www.codabench.org/competitions/10954/).

Challenge dataset: **LTDv2** [paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175339329.95323969) [dataset](https://huggingface.co/datasets/vapaau/LTDv2)

A starting kit to run some baseline experiments with YOLO is available [here](https://github.com/MarcoParola/RTIOD/tree/main/starting_kit), but feel free to reimplement from scratch all the code ;) In that case, we also release the [torch dataset](./starting_kit/src/datasets/dataset.py) to manage the dataset in the original COCO format along with some [configuration](./starting_kit/config/config.yaml), and [utility functions](./starting_kit/src/utils/).

Associated workshop at WACV2026: [Real-World Surveillance: Applications and Challenges Workshop, 6th](https://vap.aau.dk/rws/)

## Important dates
- October 17th:  Start of competition and Development Phase
- December 1st: Start of Testing Phase and end of Development Phase
- December 7th: End of competition
- December 7th: Paper submission deadline
- December 14th: Paper submission deadline (Challenge Participants)
- December 23rd: Decision notification
- January 9th 11:59 PM, PT: Camera-ready deadline
