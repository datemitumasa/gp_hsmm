# wrc_pick

##  1. <a name='Overview'></a>Overview
指定した物体IDの物体を把持するROSService

##  2. <a name='TableofContents'></a>Table of Contents
<!-- vscode-markdown-toc -->
* 1. [Overview](#Overview)
* 2. [Table of Contents](#TableofContents)
* 3. [Status](#Status)
* 4. [Quick Start](#QuickStart)
* 5. [API](#API)
* 6. [Artifacts](#Artifacts)
* 7. [Developer Information](#DeveloperInformation)
* 8. [Citations](#Citations)
* 9. [License](#License)
* 10. [Code Structure](#CodeStructure)
* 11. [Requirement](#Requirement)
* 12. [Dependences](#Dependences)
* 13. [Installation](#Installation)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  3. <a name='Status'></a>Status
2020/03/06: ROS Service作成  
2020/03/08: beta版リリース  
2020/03/10: 正式版リリース  
### TODO
- [☓] 把持確認ROSServiceの組み込み
- [ ] ROS Action 化
- [☓] Docker 化

##  4. <a name='QuickStart'></a>Quick Start

```bash
# 準備   
$ git clone http://zaku.sys.es.osaka-u.ac.jp:10080/iwata/posecnn_ros.git
$ cd wrc_pick/docker
$ ./compose_up_real.sh  
```

```bash  
# ROSServiceの使い方  
$ from wrc_pick.srv import Number2Bool, Number2BoolRequest  
$ import rospy  
$ rospy.init_node("test")  
$ srv = rospy.ServiceProxy("/wrc_pick", Number2Bool)  
$ req = Number2BoolRequest()  
# 把持させたい物体IDを入力する  
# 指定可能なDenseFusionの物体ID:[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 21]  
$ req.number = 1  
# 物体把持のIKを解いて,手先が適切に動いたかをTrue or False で返す  
# 物体把持のIKを解いて,手先が適切に動いて,物体を把持できたかをTrue or False で返す  
$ res = srv.call(req)  
$ print res.success  

```

```bash
# ROSアクションでの実行 [見送り]
```

##  5. <a name='API'></a>API
### rostopic
## Subscriber
* None  
## Publisher
* None  
### rosservice
* /wrc_pick:wrc_pick/Number2Bool  
## remap
* None  
## Parametor
* src/yaml/object_list.yaml:物体IDとそれに対応したTFの名称一覧  
* src/yaml/object_pose.yaml:物体基準座標系での把握位置を編集できる  
* src/yaml/collision.yaml:関節の禁止領域の設定(MAP依存)
* src/yaml/finger_distance.yaml:物体把持時に物体幅に合わせて開く手先の距離
* DISTANCE:物体把持時にどれだけ手前から手を近づけるかの距離  
* BACK:物体把握後にどれだけ手を下げるかの距離  
##  6. <a name='Artifacts'></a>Artifacts
- None
### 

##  7. <a name='DeveloperInformation'></a>Developer Information
- Developer: 岩田健輔
- Maintainer: 岩田健輔
- Reviewer: ???

##  8. <a name='Citations'></a>Citations
```
@inproceedings{xiang2018posecnn,
    Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
    Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
    Journal   = {Robotics: Science and Systems (RSS)},
    Year = {2018}
}
```

##  9. <a name='License'></a>License
このコードはHSR専用のコードを含むため,外部への公開は厳禁です.

---
以下オプション

##  10. <a name='CodeStructure'></a>Code Structure

##  11. <a name='Requirement'></a>Requirement

##  12. <a name='Dependences'></a>Dependences
* DenseFusion[http://zaku.sys.es.osaka-u.ac.jp:10080/iwata/dense_fusion_ros]  
* PoseCNN[http://zaku.sys.es.osaka-u.ac.jp:10080/iwata/posecnn_ros]  
* wrc_grasp_detection[http://zaku.sys.es.osaka-u.ac.jp:10080/OHMORI/wrc_grasp_detection]  
##  13. <a name='Installation'></a>Installation

Docker build時に，リポジトリのディレクトリの外に出る必要があるが，docker ignoreに他のファイルを記述しきれないため，tmpファイル内で作業する.
```bash
# How to build docker image
$ mkdir tmp
$ cd tmp
$ git clone http://zaku.sys.es.osaka-u.ac.jp:10080/iwata/wrc_pick.git
$ cd wrc_pick/docker
$ ./build.sh
```
