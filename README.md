# LCT-GAN

## Structure
```
├── datasets
│   ├── datasets.py
│   ├── __init__.py
│   ├── stft.py
│   └── tf_features.py
├── Experiments
│   ├── LCT-GAN Inference.ipynb
│   ├── LCT-GAN model.ipynb
│   ├── LCT-GAN model training.ipynb
│   ├── LCT-GAN visualization.ipynb
│   └── util.py
├── infer.py
├── __init__.py
├── losses.py
├── metrics.py
├── models
│   ├── discriminators.py
│   ├── generator.py
│   └── __init__.py
├── README.md
└── train.py
```

## Installation

```
python -m venv .venv
source .venv/bin/activate

pip install -U torchcodec
pip install pesq
pip install pystoi
```

## Datasets
We use VoiceBank_DEMAND_16k as our training data. You can download the pre-processed data directly by [link](https://drive.google.com/drive/folders/1Zg1y82NsZJbS_GQ-C1j5AJ4jGoV0MHL-?usp=sharing). The tree structure of the dataset is

```
voicebank-demand-16k
├── full
│   ├── clean_test
│   │   ├── p232_001.wav
│   │   ├── p232_002.wav
│   │   ...
│   ├── clean_train
│   │   ├── p226_001.wav
│   │   ├── p226_002.wav
│   │   ...
│   ├── noisy_test
│   │   ├── p232_001.wav
│   │   ├── p232_002.wav
│   │   ...
│   ├── noisy_train
│   │   ├── p226_001.wav
│   │   ├── p226_002.wav
│   │   ...
│   ├── test.scp
│   └── train.scp
└── sample
    ├── clean_test
    │   ├── p232_001.wav
    │   ...
    ├── clean_train
    │   ├── p226_001.wav
    │   ...
    ├── noisy_test
    │   ├── p232_001.wav
    │   ...
    ├── noisy_train
    │   ├── p226_001.wav
    │   ...
    ├── test.scp
    └── train.scp
```


## Model Training
```
# Train LCT-GAN
python train.py \
  --expr_root /content/drive/MyDrive/LCT-GAN/exprs \        # output folder
  --data_root /content/drive/MyDrive/LCT-GAN/data/voicebank-demand-16k/full \   # dataset root containing folders + scp files
  --train_scp /content/drive/MyDrive/LCT-GAN/data/voicebank-demand-16k/full/train.scp \  # train SCP file
  --test_scp  /content/drive/MyDrive/LCT-GAN/data/voicebank-demand-16k/full/test.scp \  # val/test SCP file
  --sample_rate 16000 \                                    # audio sample rate
  --segment_seconds 2.0 \                                  # training chunk length in seconds (random segments)
  --batch_size 8 \                                         # batch size
  --num_workers 8 \                                        # dataloader workers
  --epochs 1000 \                                          # total epochs
  --lr_g 2e-4 \                                            # generator/enhancer learning rate
  --lr_d 2e-4 \                                            # discriminator learning rate
  --gan_loss ls \                                          # GAN loss type: ls or hinge
  --seed 42 \                                              # random seed
  --device cuda \                                          # 'cuda' or 'cpu' (falls back to cpu if cuda unavailable)
  --log_interval 50 \                                      # print training log every N steps
  --val_interval 10 \                                      # run validation + metrics every N epochs
  --ckpt_interval 1                                        # save periodic checkpoints every N epochs
```