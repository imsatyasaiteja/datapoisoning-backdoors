# Backdoor and Poisoning Analysis

Two runnable Python scripts: targeted corner-patch backdoors, untargeted label-flipping, and robustness checks under black-box vs. white-box triggers with simple detection/sample-complexity calculations.

## Setup
- Python 3.10+, `torch`, `torchvision`, `numpy`, `matplotlib`, `facenet-pytorch`, `Pillow`.
- Recommended: run in Colab with GPU. Paths below default to the original Drive locations from the assignment; adjust as needed.
- Expected files:
  - `mnist_test_data.pt`, `model_weights_poisoned_partC.pth`
  - `CelebA_test_images.zip` extracted to `/content/images`
  - `model_weights_poisoned_partC_facenet2.tar`

## MNIST (targeted + label-flip)
`python mnist_backdoor_analysis.py --dataset-path /content/drive/MyDrive/files_updated/mnist_test_data.pt --weights-path /content/drive/MyDrive/files_updated/model_weights_poisoned_partC.pth --poison-fraction 0.05 --target-label 0 --trigger-size 3 --opt-steps 300 --opt-lr 0.003`

What it does:
- Reports clean accuracy, black-box ASR with a fixed 3x3 corner trigger at <=5% poisoning, white-box ASR with a gradient-optimized trigger, and untargeted label-flip accuracy.
- Prints Hoeffding-style sample counts needed to distinguish clean vs. poisoned performance gaps.

## FaceNet / CelebA (targeted)
1) Unzip test faces (Colab): `!unzip /content/drive/MyDrive/files_updated/CelebA_test_images.zip -d /content/`
2) Run:  
`python facenet_backdoor_analysis.py --image-folder /content/images --weights-path /content/drive/MyDrive/files_updated/model_weights_poisoned_partC_facenet2.tar --poison-fraction 0.05 --target-label 0 --trigger-size 20 --opt-steps 100 --opt-lr 0.03`

What it does:
- Evaluates black-box ASR (fixed trigger) vs. white-box ASR (optimized trigger) with <=5% poisoned samples on a subset of CelebA.
- Reports samples required to tell optimized vs. fixed triggers apart (robustness signal).

## Analysis workflow
- Keep poison rate at or below 5% to demonstrate high ASR with minimal clean accuracy drop.
- Use `--poison-fraction`, `--opt-steps`, and `--trigger-size` to probe robustness under black-box (no gradients) vs. white-box (gradient access) assumptions.
- Compare reported ASR/accuracy against defense expectations from literature (example, IEEE S&P/CCS papers); note gaps where triggers remain indistinguishable at the computed sample sizes.
