# 人体摔倒识别

#### 介绍

我们在falling posture image_datasets数据集上使用yolov5进行人体摔倒的识别，并在streamlit上进行部署，做出一个简易的界面。

对于摔倒的人识别标注为down，正常的人识别标注为person。

用户可以在图像检测、视频检测和摄像头实时检测之间进行选择。


#### 环境安装
基础环境要求
- matplotlib>=3.2.2
- numpy>=1.18.5
- opencv-python>=4.1.2
- Pillow
- PyYAML>=5.3.1
- scipy>=1.4.1
- torch>=1.7.0
- torchvision>=0.8.1
- tqdm>=4.41.0

具体的环境要求在requirements.txt中

可以在命令行输入` pip install -r requirements.txt`


#### yolov5训练过程说明

1.  将数据集放在data目录下对应的文件夹下面。并且把自己的数据集中的标注文件内容进行读取和归一化处理。
2.  对person.yaml文件中的识别类别数和类别名，train、val、test的存放位置进行更改。对于hyp.scratch.yaml的数据增强文件可以不做处理。
3.  在models中选择需要使用的yolov5模型，对相应的yaml文件中的类别数量进行修改。
4.  在train.py中对--weight（预训练权重）、--cfg（模型文件）、--data（数据集配置文件）、--epochs（训练代数）、--batch-size（批量大小）、--device（使用CPU或GPU训练）进行更改。
5.  命令行输入`python train.py`

    或者不进行第四条的更改，直接命令行内输入`python train.py --img 640 --batch 16 --epochs 300 --data ../data.yaml --cfg models/yolov5s.yaml --weights ''`
6.  训练完成后，模型权重会保存在runs/train/exp文件中，可以查看results.jpg来查看整个训练过程。


#### yolov5检测过程说明
1.  把需要检测的图像放在data/testImg中，视频放在data/testVideo中。
2.  在detect.py中的--weights更改为训练完成后的best.pt的路径，--source改为需要检测内容的目录（如果需要使用摄像头进行检测，改为0）。
3.  命令行输入`python detect.py`
4.  检测结果存放在runs/detect/exp文件中。


#### streamlit使用说明

1.  把main.py中的--weights更改为训练完成后的best.pt的路径。
2.  命令行输入`streamlit run main.py`。
3.  可以在界面的侧边栏选择图像检测、视频检测或摄像头实时检测。


#### 参考项目
- https://github.com/ultralytics/yolov5
- https://github.com/hassan-baydoun/python_final_project
