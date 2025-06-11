# 虚拟现实中高保真数字面部虚拟形象的偏振光学成像研究

## 题目: Polarized Optical Imaging for High-Fidelity Digital Facial Avatars in Virtual Reality

## 拟投：Virtual Reality & Intelligent Hardware journal



## 预解决的问题：

在人脸图像中，高光突出或者分散是一个较为常见的现象，但这不利于提取该区域的真实细节信息来进行人脸图像分解任务。常规方法要么忽略其高光特性，又或者只能依赖各种假设进行高光分离，很不符合其物理原理。



## 贡献点：

1） 采用镜面漫反射分解物理光学技术的偏振光学面部成像系统的开发。该配置包含四个支持双照明模式的光度立体（PS）相机阵列：渐变和单点恒定照明，每个相机单元生成八个渐变照明偏振图像和292个单点恒定照明偏振图像。
2） 针对获取的真实人脸图像，建立了基于微面反射的人脸皮肤材料拟合模型，为虚拟现实应用建立了高保真数字化身构建管道。该框架能够获取镜面反射漫反射分解的人脸图像和材质贴图，包括表面法线贴图、漫反射反照率贴图和镜面反射强度贴图。这些图像和贴图更准确地表示真实照明下的物理反射特性。


## data and code

step1：从百度云（链接: https://pan.baidu.com/s/1IywMivM5Av_-NHs3IhTtdQ?pwd=e8e3 提取码: e8e3）中下载已预处理数据文件夹“FaceData”放入根目录文件夹中，FaceData文件夹中包含2个被试人员的人脸数据，所有图像的分辨率都为2048*2048。另外，原始数据文件夹"RAW Images"也已经公布，所有图像的分辨率为3670*5496，百度云链接如下：https://pan.baidu.com/s/1GhQ_7R-Gjhmk3zGP5uaFgg?pwd=4p69 password：4p69

### RAW Images
表示由本系统采集的RAW图像，图像格式为.CR2，一级文件夹命名规则为：participant number and name，如01_LiangBin，二级文件夹命名规则为：PS camera number, 如Material37，三级文件命名规则为：PS camera number + sequence number，如Material37_12255。

### FaceData
表示由本系统预处理后PNG图像，图像格式为.PNG。其中，三级文件夹命名规则为：PS camera number + LED cluster number，如Material37_73，四级图像文件命名规则为：illumination pattern +polarized mode+number，如Front_Cross_L000.png, or Gradient_Cross_F.png。


step2：运行文件 ‘run_pre.py’ ，自动执行高光-漫反射图像分离、各类材质贴图拟合以及重渲染图像处理等。







