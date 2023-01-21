# fracture_MobileNetV3.py

## 概要

金属破断面の電子顕微鏡画像を読み込み、Dimple、Striation、River、Rock、Fan-Shapedのどれかを推論するプログラムです。学習はKerasライブラリのMobileNetV3モデルで行っています。

## 動作環境

[Python](https://www.python.jp/)がインストールされたパソコンで動作します。

## 使い方

- fracture_MobileNetV3.py、model.h5、labels.txtを適当なフォルダ（例えば c:\python）に置きます。
　さらに破断面の画像ファイルも適当なフォルダ（例えば、c:\data）に置きます。

- fracture_MobileNetV3.pyの15と19行目について、model.h5とlabels.txtのフルパスを編集します。
　さらに23行目について、破断面の画像ファイルのフォルダ名を編集します。

- fracture_MobileNetV3.pyを実行します。不足分のライブラリがあればインストールしてください。


## ご利用に関して

使用結果について当方は責任は負いません。

## 開発環境

- Windows11
- VSC 1.7.3.1
- Python 3.8.10
- numpy 1.20.0
- keras 2.11.0
- pillow 9.3.0
