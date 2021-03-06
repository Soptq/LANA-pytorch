# LANA-pytorch
Official Pytorch implementation of "LANA: Towards Personalized Deep Knowledge Tracing Through Distinguishable Interactive Sequences"

![](figures/kt-example.png)

## Abstract

> In educational applications, Knowledge Tracing (KT), the problem of accurately predicting students' responses to future questions by summarizing their knowledge states, has been widely studied for decades as it is considered a fundamental task towards adaptive online learning. Among all the proposed KT methods, Deep Knowledge Tracing (DKT) and its variants are by far the most effective ones due to the high flexibility of the neural network. However, DKT often ignores the inherent differences between students (e.g. memory skills, reasoning skills, ...), averaging the performances of all students, leading to the lack of personalization, and therefore was considered insufficient for adaptive learning. To alleviate this problem, in this paper, we proposed Leveled Attentive KNowledge TrAcing (LANA), which firstly uses a novel student-related features extractor (SRFE) to distill students' unique inherent properties from their respective interactive sequences. Secondly, the pivot module was utilized to dynamically reconstruct the decoder of the neural network on attention of the extracted features, successfully distinguishing the performance between students over time. Moreover, inspired by Item Response Theory (IRT), the interpretable Rasch model was used to cluster students by their ability levels, and thereby utilizing leveled learning to assign different encoders to different groups of students. With pivot module reconstructed the decoder for individual students and leveled learning specialized encoders for groups, personalized DKT was achieved. Extensive experiments conducted on two real-world large-scale datasets demonstrated that our proposed LANA improves the AUC score by at least 1.00% (i.e. EdNet 1.46% and RAIEd2020 1.00%), substantially surpassing the other State-Of-The-Art KT methods.

## Main Architecture

![](figures/lana-arch.png)

![](figures/leveled-learning.png)

## Quickstart
### Cloning
```
git clone https://github.com/Soptq/LANA-pytorch
cd LANA-pytorch
```

### Installation
```
python depen_install.py
```

### Dataset Preparation

you need to manually download the dataset, and perform preprocessing on it. A sample preprocessing script is provided as `sample_data_preprocess.py`

Note that if you are going to try Leveled Learning, you must pass `--irt` to the preprocessing script.

### Run LANA
```
python main.py -d YOUR_PREPROCESSED_DATA
```

configurations of the experiments are set in `config.py`.

After training to convergence, you can further improve the performance of it by applying Leveled Learning (optional):

```
python main_ll -d YOUR_PREPROCESSED_DATA -m YOUR_TRAINED_MODEL ...other arguments
```

## Results

| Dataset  | Model | AUC |
| ------------- | ------------- | ------------- |
| EdNet  | DKT  | 0.7638 |
| EdNet  | DKVMN  | 0.7668 |
| EdNet  | SAKT  | 0.7663 |
| EdNet  | SAINT  | 0.7816 |
| EdNet  | SAINT+  | _0.7913_ |
| EdNet  | SAINT+ & BM  | 0.7935 |
| EdNet  | LANA  | **0.8059** |

## Cite

```
@inproceedings{zhou2021lana,
  title={LANA: Towards Personalized Deep Knowledge Tracing Through Distinguishable Interactive Sequences},
  author={Yuhao Zhou, Xihua Li, Yunbo Cao, Xuemin Zhao, Qing Ye, Jiancheng Lv},
  organization={Sichuan University, Tencent Inc.}
  year={2021}
}
```

## License

```
MIT License

Copyright (c) 2021 Soptq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
