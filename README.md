# rendering

## 動作環境まとめ
- tensorflow 1.15はpython<=3.7に対応
- pyglet(pyrender, trimeshの依存先)の最新版は>=3.8に対応
- pytorch3dは>=3.8に対応
- rednerは>=3.6に対応

対応策
1. rednerを使用する．
2. pygletではなくosmesaかeglを用いてpyrenderなどでレンダリング (python3.7)
3. tensorflow 2以降に対応できるようにmanipnetのコードを変換 (python3.8)
4. pytorchなどでmanipnetを追実装（end2endで学習できるようにすることを考えるとこれがベストかもしれない）

||python|3.7|3.8|
|---|---|---|---|
|tensorflow1.15|<=3.7|o|x|
|tensorflow2|3.6～3.9|o|o|
|redner|>=3.6|||
|pytorch3d|>=3.8|x|o|
|trimesh(interactive)|>=3.7|o|o|
|trimesh(pyglet)|>=3.8|x|o|
|pyrender|>=3.8|x|o|
|pyglet|>=3.8|x|o|
|pyglet==1.5.28|>=3.6|o|o|
|bpy4.0.0|==3.10|x|x|

<!-- |pytorch1.12.0|>=3.7, <=3.10||CUDA 11.3, CUDNN 8.3.2.44 -->


## Installation
- git cloneあるいはsubmoduleの更新 (manipnetはsubmoduleで追加)
```
git clone *.git --recursive
#or
git submodule update --init --recursive
```
- (docker立てる)
- condaのインストール
```
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda # condaのinstall
```

- manoデータのダウンロード
```
data/mano_v1_2/models/mano
 |- MANO_LEFT.pkl
 |- MANO_RIGHT.pkl
```

### python3.7の環境
- tensorflow1.15でmanipnetを動かせる環境
- rednerを用いてレンダリング
```
conda create -n render-hand python=3.7
source /opt/conda/etc/profile.d/conda.sh
conda activate render-hand
```
- pip install
```
pip install smplx scipy chumpy trimesh plotly matplotlib
```
- python>=3.7をサポートしているpythorch=1.12をインストール (https://pytorch.org/blog/deprecation-cuda-python-support/)
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
- renderをインストール
```
pip install --upgrade redner-gpu
```
- tensorflow==1.15をインストール
```
pip install tensorflow==1.15 protobuf==3.20
```


### python3.8の環境
- pytorch3dを用いてmanoとオブジェクトをレンダリングできる環境
- 可能ならばtf2へのコンバートもおこなう

```
conda create -n py3.8 python=3.8 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate py3.8
```
- pytorchとpytorch3dのinstall
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
- その他のライブラリ
```
pip install smplx scipy chumpy trimesh plotly matplotlib
```

### tf1からtf2への変換
```
tf_upgrade_v2 \
  --intree my_project/ \
  --outtree my_project_v2/ \
  --reportfile report.txt
```