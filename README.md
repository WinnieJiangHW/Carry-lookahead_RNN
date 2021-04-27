##  Sequence Modeling Tests on Carry-lookahead RNN (CL-RNN)

This repository contains the experiments done in the work ***Recurrent Neural Network from Adder’s Perspective: Carry-lookahead RNN*** by Haowei Jiang and Feiwei Qin.

We performed an ablation experiment on Sequential MNIST and comparative experiments on music and language modeling. According to the results, we verify the effectiveness of CL-RNN.

### Repository Structure

The repository consists of experiment code, results and pre-trained models.

The code is in the [Benchmarks](https://github.com/WinnieJiangHW/TCN_VRN/tree/main/Benchmarks) directory and contains the benchmarks to the following tasks:

- **Sequential MNIST**: digit classification
- **JSB Chorales**: polyphonic music
- **Nottingham**: polyphonic music
- **PennTreebank**: char-level language model
- **text8**: char-level language model

> The default model in poly_music and char_cnn is CL-RNN (model proposed in the paper), and other comparison models are respectively placed in [Test_results](https://github.com/WinnieJiangHW/TCN_VRN/tree/main/Test_results) directory.

There are also **experiment results** and **pre-trained models** placed in [Test_results](https://github.com/WinnieJiangHW/TCN_VRN/tree/main/Test_results) directory.

### Runtime Environment

1. The code should be directly runnable with PyTorch v1.0.0 or above.
2. We train the models with PyTorch v1.7.1 and the pre-trained models can’t be load with PyTorch v1.8 or above.

### Acknowledgements

The code framework is based on https://github.com/locuslab/TCN.