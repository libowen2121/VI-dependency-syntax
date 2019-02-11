# Code for 'Dependency Grammar Induction with a Neural Variational Transition-based Parser' (AAAI2019)

## Preprocessing: 

[Brown Clustering](https://github.com/percyliang/brown-cluster) <br>
After clustering, add extra two fields (cluster index and token index inside the cluster) to the UD/WSJ dataset <br>
[Customized TorchText `0.2.3`](https://drive.google.com/file/d/1sDGfPwq0vNwSh-2JmXomi8GyTjAvOY81/view?usp=sharing)<br>

*Since WSJ corpus is not publicly available, training and evaluating scripts for UD are as below*.

## Supervised training (for UD)
Train the encoder `./ud_scripts/ud_train_encoder.sh`  
Train the decoder `./ud_scripts/ud_train_decoder.sh`  
Note: <br>
&nbsp;&nbsp;&nbsp;&nbsp;Set no length limitation for preprocessing to keep a full vocabulary; <br>
&nbsp;&nbsp;&nbsp;&nbsp;Set random seed to be -1

## Weakly-/Un-supervised training (for UD)
Rule settings:  
&nbsp;&nbsp;&nbsp;&nbsp;Universal Ruels: `--pr_fname "./data/pr_rules/ud_c/"$LANGUAGE"_0.5.txt"`  
&nbsp;&nbsp;&nbsp;&nbsp;Weakly Supervised: `--pr_fname "./data/pr_rules/ud_c/"$LANGUAGE"_10_gt.txt"` 

Pretrain: `cd ud_scripts && ./ud_pre.sh`  
Finetune: `cd ud_scripts && ./ud_ft.sh`

## Evaluation (for UD)
`cd ud_scripts && ./ud_test.sh`