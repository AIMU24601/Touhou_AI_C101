# Touhou_AI_C101

C101にて発表予定の東方Project自動攻略AIです。

予定
・Neuroevolution
　CNE　NEAT　HYPERNEAT　DQNの比較？
　RNNも使えるようだが少し難しそうなので検討
　
・強化学習
　Neuroevolutionと比較してRNNとかを使ってDQNをやる
　PyTorch使う

・他の研究と組み合わせる
　弾幕をラベルするおもしろそうな研究があるのでこれもやってみる
　https://ieee-cog.org/2021/assets/papers/paper_312.pdf

参考にしたサイト
PyTorchでGPUの動かすために必要
Windows10にPyTorch1.10とCUDA11.3の環境を作る
　https://zenn.dev/opamp/articles/c5e200c6b75912

最低限の画像分類
【PyTorch】MNISTのサンプルプログラム
　https://imagingsolution.net/deep-learning/pytorch/pytorch_mnist_sample_program/

画像の正規化
【Ptyorch】ToTenserした画像をNormalizationすることに意味はあるのでしょうか
　https://teratail.com/questions/234027

入力の次元やone-hot-vectorについて
Kerasを勉強した後にPyTorchを勉強して躓いたこと
　https://takuroooooo.hatenablog.com/entry/2020/10/30/Keras%E3%82%92%E5%8B%89%E5%BC%B7%E3%81%97%E3%81%9F%E5%BE%8C%E3%81%ABPyTorch%E3%82%92%E5%8B%89%E5%BC%B7%E3%81%97%E3%81%A6%E8%BA%93%E3%81%84%E3%81%9F%E3%81%93%E3%81%A8

画像のパディング
OpenCVでzero-paddingを1行でする方法
　https://qiita.com/sota0726/items/540d3be3e570cbca644e

numpyのパディング
numpy.pad関数完全理解
　https://qiita.com/kuroitu/items/51f4c867c8a44de739ec

Datasetの作り方
PyTorch: DatasetとDataLoader (画像処理タスク編)
　https://ohke.hateblo.jp/entry/2019/12/28/230000

segmentation_models_pytorchを使ったセマンティックセグメンテーション
PyTorchによるMulticlass Segmentation - 車載カメラ画像のマルチクラスセグメンテーションについて．
https://qiita.com/nigo1973/items/c62578fccc7230ba48f8

画像分類・物体検出・セグメンテーションの比較
　https://data-analysis-stats.jp/%E6%B7%B1%E5%B1%9E%E5%AD%A6%E7%BF%92/%E7%94%BB%E5%83%8F%E5%88%86%E9%A1%9E%E3%83%BB%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%BB%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E6%AF%94%E8%BC%83/

