# gp_hsmm

##  1. <a name='Overview'></a>Overview
GP-HSMMは連続な情報を類似した系列ごとに分節分類を行うモデルである.  
本モデルはGP-HSMMに物体情報を付加することで,物体操作時のエンドエフェクターの軌道の分節に特化したモデルである.  
参照点という軌道の特徴を強調する座標系を設計することで,より解釈性の高い基本系列の獲得を可能としている.
![Fruture](https://user-images.githubusercontent.com/28037675/76589881-e8642680-652e-11ea-8d23-433b39b0c36f.png)
##  2. <a name='TableofContents'></a>Table of Contents
<!-- vscode-markdown-toc -->
* 1. [Overview](#Overview)
* 2. [Table of Contents](#TableofContents)
* 3. [Requirements](#Requirements)
* 4. [Code Structure](#CodeStructure)
* 5. [Datasets](#Datasets)
* 6. [Training](#Training)
* 7. [Results](#Training)
* 8. [Generate Trajectory](#GenerateTrajectory)
* 9. [Tips for your own dataset](#TipsforyourownDataset)
* 10. [Citations](#Citations)
* 11. [License](#Requirement)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  3. <a name='Requirements'></a>Requirements
- Python 2.7
- ROS indigo/kinetic/melodic
- cython
- numpy
- matplotlib
- scipy
- graphviz
- pandas
## 4. <a name='Code Structure'></a>Code Structure  
- learn
    - learn/GaussianProcess.pyx : Cythonコーティングのガウス過程
    - learn/GaussianProcess.so : コンパイルされたCythonのガウス過程
    - learn/GaussianProcessMulitiDim.py : ガウス過程を多次元に展開するコード
    - learn/RPGPHSMMs.py : RP-GP-HSMMのメインコード
    - learn/RP.py : RP-GP-HSMMの実行コード
    - learn/RPOD.py : RPOD-GP-HSMMに拡張して学習する実行コード
    - data
        - data/object_data.csv : RP or RPOD-GP-HSMMを学習する際の参照点の候補
        - data/continuous/data{0:d}.csv : 学習する連続情報(サンプル)
- config
    - config/gp_hsmm_parameter.yaml : GP-HSMMで学習するための設定ファイル
##  5. <a name='Datasets'></a>Datasets
csv 形式の連続情報のデータを与え,設定ファイルから次元数を与えることで学習することが可能である.  
参照点(RP)を用いて学習を行うためには,与える情報は7次元の姿勢情報である必要がある.  
姿勢情報に情報を追加して学習する場合は,８次元目以降に設定することで学習が可能である.  
姿勢情報以外を学習する場合は後述の[Training](#Training)で指定する設定が必要となる.
### 

##  6. <a name='Training'></a>Training
学習を行う場合は,まず連続情報に合わせた設定ファイルの作成が必要になる.
- yaml/gp_hsmm_parameter.yaml :  
    - gp_hsmm_parameter :  
        - data_dimention : 学習する情報の次元数  
        - time_thred : 分節に適用する物体情報の時間の許容差分(sec)  
        - distance_thread : 分節に適用する物体と軌道との距離の閾値(m),time_thread以内に閾値以内に物体が近づくことがなければ学習に適用されない  
        - max_distance_thread : 分節に適用する物体と手先との限界距離(m),これ以上離れた物体は学習に適用されない  
        - data_time_sparse : 学習データの時間幅(sec),設定した値より時間幅が小さい場合は自動で調節する  
        - average_length : 分節される軌道の平均長  
        - min_length : 分節される軌道の最低長  
        - max_length : 分節される軌道の最高長  
        - landmark_setting : 分類されるクラス数,それぞれの参照点ごとに数を指定
            - own_landmark_class : 自身を基準とした軌道の変換を行う参照点.物体情報を必要としない
            - z_axis_rotate_class : 物体のz軸(鉛直方向)を回転させ,物体の軸を軌道に向ける変換を行う参照点.物体情報を必要とする
            - z_and_y_axis_rotate_class : 物体のz軸とy軸(左右)を回転させ,物体の軸を軌道に向ける変換を行う参照点.物体情報を必要とする
            - no_rotate_class : 物体固有の座標系に従い,物体の軸から見た軌道に変換を行う参照点.物体情報を必要とする.
            - no_landmark_class : 参照点を使わず,７次元姿勢情報以外を学習する場合はこのClass以外を0とする.
        - object_csvdata: 参照点候補を格納したcsvのパス
        - category : 学習に用いる物体の名前
        - continuous_csvdata : 学習に用いる連続情報のcsvデータ
        - continuous_data_name : 学習に用いる連続情報の名前
  
```bash
# RP-GP-HSMMの学習  
$ cd learn
$ python RP.py
```  

```bash
# RP-GP-HSMMの学習  
$ cd learn
$ python RPOD.py
```  

##  7. <a name='Results'></a>Results
- learn
    - learn/save
        - learn/save/category
        - learn/save/category/GP_m{0:d}.csv : 学習されたガウス過程の平均
        - learn/save/category/GP_sigma{0:d}.csv : 学習されたガウス過程の分散
        - learn/save/category/class{0:3d}.npy : クラスごとに分類された分節データ
        - learn/save/category/segm{0:03d}.txt : 学習データの時間ごとの分類結果
        - learn/save/category/slen{0:03d}.txt : 学習データの分節ごとの分類結果
        - learn/save/category/stamps{0:03d}.txt : 学習データの分節ごとの時刻
        - learn/save/category/trans.npy : 学習された分類ごとの遷移確率
        - learn/save/category/trans_bos.npy : 学習された分類に対する初期状態からの遷移確率
        - learn/save/category/trans_eos.npy : 学習された分類ごとの終了状態への遷移確率
        - learn/save/category/trans.png : 学習された分類ごとの図示された遷移確率
        - learn/save/category/test.svg : 学習された分類の時間ごとのガウス過程の平均と分散
        - learn/save/category/class.png : 学習された分類ごとの分節された軌道

##  8. <a name='GenerateTrajectory'></a>Generate Trajectory
* [comming soon]
##  9. <a name='Citations'></a>Citations
* [comming soon]
##  10. <a name='License'></a>License
* [comming soon]