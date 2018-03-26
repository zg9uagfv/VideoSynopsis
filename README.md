# VideoSynopsis

## 环境
1.  ubuntu16.04 
2.  cmake version 3.10.2
3.  opencv-3.4.0
4.  dlib-19.9
5.  cuda8.0 cudnn6


## 介绍
借鉴开源的项目的代码(记不清具体名字),采用SSD对视频中的目标检测和跟踪(粒子或卡尔曼)，采用泊松融合把每个跟踪目标叠到背景中．

注意:
1.  SSD的模型文件可以到https://github.com/weiliu89/caffe/tree/ssd进行下载.
