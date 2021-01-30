# 概要  
本プログラムは、音声ファイルを解析して音素を取得し、音声に合致した口パクを表示することを目的としている。  
音声から音素を抽出する方法として、深層畳み込みニューラルネットワークを使用した。  
ネットワークにスペクトログラムと対応する音素ラベルを与えて分類学習を行い、スペクトログラムから音素を予測するモデルを学習する。  
推論時にはスペクトログラムから音素を予測する。音声を再生するとともに口パク画像を表示する事が可能。  

## 環境  
実行確認環境  
```
Windows10
Python 3.6
```
本コードはJVS corpusを用いて学習するため、以下のリンクからデータセットのダウンロードが必要  
https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus  
その他のPythonパッケージとして、以下が必要  
```
PyTorch
scipy
joblib
PyAudio
Pygame
matplotlib
squib (https://github.com/pstuvwx/squib)
```

## 使用方法  
JVS corpusをダウンロードし、以下のディレクトリ構造であることを前提とする  
```
workdir/
  |-Lip/
  |  |-dataset_jvs.py
  |  |-demonstrate.py
  |  | ...
  |
  |-jvs_ver1/
     |-jvs001/
     |  |-falset10/
     |  | ...
     |
     |-jvs002/
     | ...
     |-jvs100/
```

### 学習  
`trainer.py`を実行する。  
```
python Lip/trainer.py jvs_ver1
```
引数は以下の通り。  
#### dir_jvs  
必須。JVS corpusのディレクトリ  
#### n_win  
STFTの窓幅  
#### n_hop  
STFTの移動幅  
#### n_class  
口形状クラス数  
#### n_unit  
CNN中間層のチャネル数  
#### n_layer  
CNNの中間ResBlock数  
#### ksize  
CNNの畳み込みカーネルサイズ  
#### n_frames  
学習時に連続的に入力するフレーム数  
#### n_skip  
データセットのスキップ量  
CPUメモリ少ない場合、この値を大きくすることで、データセットの1/n_skipだけ利用する
#### lr  
学習係数  
#### eps  
STFTの対数演算時のゼロ割防止用微小数  
#### batch_size  
ミニバッチサイズ  
#### n_epoch  
学習エポック数  
#### device  
学習を実行するデバイス  
#### dst  
出力先  

### 推論  
`demonstrate.py`を実行する。  
```
python Lip/demonstrate.py jvs_ver1/jvs001/nonpara30/wav24kHz16bit/BASIC5000_0025.wav result/models/models_60.pth
```
引数は以下の通り。  
#### wavfile  
音声ファイルのパス  
#### model  
学習済みモデルのパス  
学習曲線を見て、`val/loss`が最小のモデルを使用する  
#### n_win  
STFTの窓幅  
学習時と一致させること  
#### n_hop  
STFTの移動幅  
学習時と一致させること  
#### n_class  
口形状クラス数  
学習時と一致させること  
#### n_unit  
CNN中間層のチャネル数  
学習時と一致させること  
#### n_layer  
CNNの中間ResBlock数  
学習時と一致させること  
#### ksize  
CNNの畳み込みカーネルサイズ  
学習時と一致させること  
#### eps  
STFTの対数演算時のゼロ割防止用微小数  
学習時と一致させること  
#### device  
推論を実行するデバイス  
#### k_median  
平滑化フィルタのカーネルサイズ  
#### dir_img  
口パク画像が存在するディレクトリ  

### JVS ccorpus以外のデータを学習  
`dataset_jvs.py`を適切に変更すること  
 - `phoneme2index`を編集し、音素と口形状の対応関係を記述する。指定した数値`i`が口パク画像`i.png`と対応して表示される  
 - 音素ラベルがJuliusと異なる形式の場合、`load_lab`を編集し、波形のサンプル点ごとの音素ラベルを返す関数にすること
