# LH-Mix

This work has been accepted as the long paper ''LH-Mix: Local Hierarchy Correlation Guided Mixup over Hierarchical Prompt Tuning'' in KDD 2025.

## Data Preprocess:

The code is based on "Implement of HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification" . We appreciate their sharing.

Please refer to https://github.com/wzh9969/HPT.

## Run:

You can start LH-Mix directly by running the following code:

```
python train.py --name public --batch 16 --data rcv1 --threshold 0.75 --alpha 1 --warm_up 0
```

## Cite:

```
@article{kong2024lhmix,
	title={LH-Mix: Local Hierarchy Correlation Guided Mixup over Hierarchical Prompt Tuning},
	author={Fanshuang Kong, Richong Zhang and Ziqiao Wang},
	booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
	year={2025},
}
```
