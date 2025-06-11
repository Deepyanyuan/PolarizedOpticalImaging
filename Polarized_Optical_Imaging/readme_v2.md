# Polarized Optical Imaging for High-Fidelity Digital Facial Avatars in Virtual Reality

## title: Polarized Optical Imaging for High-Fidelity Digital Facial Avatars in Virtual Reality

## Proposed investment：Virtual Reality & Intelligent Hardware journal



## Pre-solved problems：

In face images, highlighting or specular is a relatively common phenomenon, but this is not conducive to extracting the real detail information  for the task of facial image decomposing. Conventional methods either ignore its specular characteristics, or can only rely on various assumptions for specular separation, which does not conform to its physical principles. 


## Contributions：

1) Development of a polarization-optical facial imaging system employing physical optics techniques for specular-diffuse decomposition. The configuration incorporates four photometric stereo (PS) camera arrays supporting dual illumination modalities: gradient-ramp and single-point constant illumination, yielding eight gradient-ramp illumination polarized images and 292 single-point constant illumination polarized images per camera unit.
2) A microfacet reflection-based facial skin material fitting model was developed for acquired real-world facial images, establishing a high-fidelity digital avatar construction pipeline for virtual reality applications. This framework enables the acquisition of specular-diffuse decomposed facial imagery and material maps that more accurately represent physical reflectance properties under realistic illumination. The generated material suite includes surface normal map, diffuse albedo map, and specular intensity map.


## data and code


Step1: download the preprocessed data folder "FaceData" from Baidu Cloud (links：https://pan.baidu.com/s/1IywMivM5Av_-NHs3IhTtdQ?pwd=e8e3 
password：e8e3)  and put it into the root folder. The folder contains the face data of two subjects, and the resolution of all images is 2048*2048. In addition, the original data folder "RAW Images" has also been published, and the resolution of all images is 3670*5496. Baidu Cloud link is as follows: links：https://pan.baidu.com/s/1GhQ_7R-Gjhmk3zGP5uaFgg?pwd=4p69 password：4p69


### RAW Images
It represents raw images collected by our system in the format of CR2， The naming rules of the first level folder are: Participant number and name, such as 01_liangbin, the naming rules of the second level folder are: PS camera number, such as material37, and the naming rules of the third level file are: PS camera number+sequence number, such as material37_12255.
### FaceData
It represents the image preprocessed by our system, and the image format is PNG。 Among them, the naming rules of the three-level folder are: PS camera number+LED cluster number, such as material37_73, and the naming rules of the four-level image file are: illumination pattern+polarized mode+number, such as front_cross_l000.png, or gradient_cross_f.png.


Step2: run the file 'run_pre.py' to automatically perform specular-diffuse image separation, various material maps fitting and re-rendered image processing.








