# -*- encoding: utf-8 -*-
'''
@File    : processor.py
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
'''
'''
代码描述

'''
# ----- Load libraries -------------------------------------
import imageio
from imageio.core import util
from numpy.core.fromnumeric import sort
import torch
import os
import glob
import cv2
import yaml
import math
import time
import numpy as np
from numba import jit
from . import utils_real as utils
from tqdm import tqdm
import matplotlib.pyplot as plt



# ---- Custom functions -----------------------------------

class Processor():
    def __init__(self, cfgs):
        self.isSave = cfgs.get('isSave', False)
        self.isMax = cfgs.get('isMax', False)
        self.isGamma = cfgs.get('isGamma', False)
        self.isName = cfgs.get('isName', False)
        self.root_path = cfgs.get('root_path', 'E:/@Datas/Lightstage')
        self.path_views = cfgs.get('path_views', '0_73')
        self.start = cfgs.get('start', 0)
        self.end = cfgs.get('end', 1)

        ## parameters
        self.r0 = cfgs.get('r0', 0.02549)
        self.darkThreshold = cfgs.get('darkThreshold', 0.9)
        self.maxAngle = cfgs.get('maxAngle', 0.3925)
        self.gauss_kernel_size = cfgs.get('gauss_kernel_size', 13)
        self.sigma_magnification = cfgs.get('sigma_magnification', 1)
        self.Fit_Counter = cfgs.get('Fit_Counter', 0)
        self.temp_ss = cfgs.get('temp_ss', 0.45)
        self.weight_mask = cfgs.get('weight_mask', 0.15)
        self.weight_normal = cfgs.get('weight_normal', 0.2)
        self.weight_lambert = cfgs.get('weight_lambert', 0.2)
        self.weight_shading = cfgs.get('weight_shading', 0.2)
        self.mix = cfgs.get('mix', 0.5)
        self.rho_dt_num = cfgs.get('rho_dt_num', 160)
        self.lightIntensity = cfgs.get('lightIntensity', 70)
        self.diffuseIntensity = cfgs.get('diffuseIntensity', 0.8)
        self.ambientIntensity = cfgs.get('ambientIntensity', 0.1)
        self.gamma = cfgs.get('gamma', 2.2)

        self.k_ss = cfgs.get('k_ss', 0.1)
        self.resize = cfgs.get('resize', None)
        self.flag = cfgs.get('flag', 3)

        ## calculate imgs path
        self.path_lightDirsFront = cfgs.get('path_lightDirsFront', 'lightDirsFront.npy')
        self.path_lightDirs = cfgs.get('path_lightDirs', 'lightDirs.npy')
        self.path_viewDirs = cfgs.get('path_viewDirs', 'viewDirs.npy')
        self.path_imgs_Full_diffuse = cfgs.get('path_imgs_Full_diffuse', 'imgs_Full_diffuse_4whc.npy')
        self.path_imgs_Full_residue = cfgs.get('path_imgs_Full_residue', 'imgs_Full_residue_4whc.npy')
        self.path_imgs_Full_single = cfgs.get('path_imgs_Full_single', 'imgs_Full_single_4whc.npy')
        self.path_img_front_residue = cfgs.get('path_img_front_residue', 'img_Front_residue_whc.npy')
        self.path_img_front_diffuse = cfgs.get('path_img_front_diffuse', 'img_Front_diffuse_whc.npy')
        self.path_img_front_single = cfgs.get('path_img_front_single', 'img_Front_single_whc.npy')
        self.path_img_full_residue = cfgs.get('path_img_full_residue', 'img_Full_residue_whc.npy')
        self.path_img_full_diffuse = cfgs.get('path_img_full_diffuse', 'img_Full_diffuse_whc.npy')
        self.path_img_full_single = cfgs.get('path_img_full_single', 'img_Full_single_whc.npy')
        
        ## calculate maps path
        self.path_map_normal = cfgs.get('path_map_normal', 'map_normal.npy')
        self.path_map_mask = cfgs.get('path_map_mask', 'map_mask.npy')
        self.path_map_specular = cfgs.get('path_map_specular', 'map_specular.npy')
        self.path_map_sscatter = cfgs.get('path_map_sscatter', 'map_sscatter.npy')
        self.path_map_diffuse = cfgs.get('path_map_diffuse', 'map_diffuse.npy')
        self.path_result = cfgs.get('path_result', 'result.npy')
        
        ## render kinds of component
        self.path_render_unlit_shading = cfgs.get('path_render_unlit_shading', 'render_unlit_shading.npy')
        self.path_render_rho_dt_Ls = cfgs.get('path_render_rho_dt_Ls', 'render_rho_dt_Ls.npy')
        self.path_render_rho_dt_V = cfgs.get('path_render_rho_dt_V', 'render_rho_dt_V.npy')
        self.path_render_unlit_mix_irradiance = cfgs.get('path_render_unlit_mix_irradiance', 'render_unlit_mix_irradiance.npy')
        self.path_render_unlit_shading = cfgs.get('path_render_unlit_shading', 'render_unlit_shading.npy')
        self.path_render_unlit_specular = cfgs.get('path_render_unlit_specular', 'render_unlit_specular.npy')
        self.path_render_unlit_sscatter = cfgs.get('path_render_unlit_sscatter', 'render_unlit_sscatter.npy')
        self.path_render_unlit_subscatter = cfgs.get('path_render_unlit_subscatter', 'render_unlit_subscatter.npy')
        self.path_render_unlit_diffuse = cfgs.get('path_render_unlit_diffuse', 'render_unlit_diffuse.npy')
        self.path_render_unlit_diffimage = cfgs.get('path_render_unlit_diffimage', 'render_unlit_diffimage.npy')
        self.path_render_unlit_singleimage = cfgs.get('path_render_unlit_singleimage', 'render_unlit_singleimage.npy')

        ## images2video
        self.video_fps = cfgs.get('video_fps', 12)
        self.video_suffix = cfgs.get('video_suffix', '.mp4')


    def createPath(self):
        # 获取该目录下的所有文件或文件夹目录路径
        root_path = self.root_path
        files = glob.glob(root_path + '/*')
        self.objects_path = []
        for file in files:
            # 判断该路径下是否是文件夹
            if (os.path.isdir(file)):
                # 分成路径和文件的二元元组
                path_object = os.path.split(file)
                # print('获得不同的人物模型的文件：',path_object[1])
                self.objects_path.append(path_object[1])

                # 创建文件夹
                # print('path_views:', self.path_views)
                utils.createSaveFolders(file, self.path_views)

        return self.objects_path


    def ALL(self, path_object):
        path_root_object = os.path.join(self.root_path, path_object)
        # for v in tqdm(range(len(self.path_views))):
        for v in tqdm(range(self.start, self.end)):
            path_view = 'Material' + str(self.path_views[v])
            print('path_view', path_view)

            print('\n')
            print('------------Step 00: initial works-----------')
            print('分离图像 specular-diffuse separation...')
            self.X00_separateImage(path_root_object, path_view)
            
            print('\n')
            print('------------Step 01: calculate kinds of albedos-----------')
            print('计算法线和遮罩 normal and mask...')
            self.X01_calculateNormalandMask(path_root_object, path_view)

            print('计算高光反射贴图 map_specular ...')
            self.X01_calculateSpecularAlbedo(path_root_object, path_view)
            
            print('计算单次散射贴图 map_sscatter ...')
            self.X01_calculateSScatterAlbedo(path_root_object, path_view)
            
            print('计算颜色贴图 map_diffuse ...')
            self.X01_calculateDiffuseAlbedo(path_root_object, path_view)
            
            print('计算前点照明图像 ...')
            self.X01_renderFrontlitImage(path_root_object, path_view)
            
            print('渲染透射率图像 irradiance...')
            self.X01_renderTransmittance(path_root_object, path_view)
            
            
            print('\n')
            print('------------Step 02: render kinds of components-----------')
            print('渲染单光源阴影 shading...')
            self.X02_renderShade(path_root_object, path_view)
            
            print('渲染单光源混合辐射度图像 mix_irradiance...')
            self.X02_renderMixIrradiance(path_root_object, path_view)
            
            print('渲染单光源单次散射分量 component_sscatter...')
            self.X02_renderSScatterComponent(path_root_object, path_view)
            
            print('渲染单光源高光分量 component_specular...')
            self.X02_renderSpecularComponent(path_root_object, path_view)
            
            print('渲染漫反射分量 component_diffuse...')
            self.X02_renderDiffuseComponent(path_root_object, path_view)
            
          


    def X00_separateImage(self, path_root_object, path_view):
        # 合并同一个人物模型的多个目录
        # 获取该目录下的所有文件或文件夹目录路径
        start_X00 = time.time()

        path_view = str(path_view)
        # 记录相机数目,以及相机正前方光源的编号
        fields = path_view.split('_')
        print('fields', fields)
        
        path_root_view = os.path.join(path_root_object, 'Source/' + path_view)
        print('path_root_view',path_root_view)
        
        

        ## 分离梯度照明图像
        files_full_cross = sorted(glob.glob(path_root_view + '/Gradient_Cross_*'))
        files_full_parallel = sorted(glob.glob(path_root_view + '/Gradient_Parallel_*'))
        # print('files_full_cross',files_full_cross)
        # print('files_full_parallel',files_full_parallel)
        imgs_full_diffuse = []
        imgs_full_residue = []
        imgs_full_single = []
        for gi in range(len(files_full_cross)):
            srcCrossPath = files_full_cross[gi]
            srcParallelPath = files_full_parallel[gi]
            imgCross = utils.readPNG(srcCrossPath, self.resize, self.gamma)
            imgParallel = utils.readPNG(srcParallelPath, self.resize, self.gamma)
            imgDiffuse, imgResidue, imgSingle = utils.splitSingleDiffuseSpecular(imgCross, imgParallel)
            imgSingle_1 = imgDiffuse + imgResidue
            imgs_full_diffuse.append(imgDiffuse)
            imgs_full_residue.append(imgResidue)
            # imgs_full_single.append(imgSingle)
            imgs_full_single.append(imgSingle_1)
            
            if self.isSave:
                path_png = path_root_object + '/Show/image_' + path_view
                imgDiffuseName = os.path.split(srcCrossPath)[1].replace('Cross', 'Diffuse')
                imgResidueName = os.path.split(srcParallelPath)[1].replace('Parallel', 'Residue')
                imgSingelName = os.path.split(srcCrossPath)[1].replace('Cross', 'Single')
                # imgSingelName_1 = os.path.split(srcCrossPath)[1].replace('Cross', 'Single_1')
                imgDiffuseNameFile = os.path.splitext(imgDiffuseName)[0]
                imgResidueNameFile = os.path.splitext(imgResidueName)[0]
                imgSingelNameFile = os.path.splitext(imgSingelName)[0]
                # imgSingelNameFile = os.path.splitext(imgSingelName_1)[0]
                imgDiffuse = np.power(imgDiffuse, 1/self.gamma)
                imgResidue = np.power(imgResidue, 1/self.gamma)
                # imgSingle = np.power(imgSingle, 1/self.gamma)
                imgSingle = np.power(imgSingle_1, 1/self.gamma)
                if self.isMax:
                    utils.savePNG(path_png + '/' + imgDiffuseName,imgDiffuse/imgDiffuse.max(), imgDiffuseNameFile,self.isName)
                    utils.savePNG(path_png + '/' + imgResidueName,imgResidue/imgResidue.max(), imgResidueNameFile,self.isName)
                    utils.savePNG(path_png + '/' + imgSingelName,imgSingle/imgSingle.max(), imgSingelNameFile,self.isName)
                    # utils.savePNG(path_png + '/' + imgSingelName_1,imgSingle_1/imgSingle_1.max(), imgSingelNameFile_1,self.isName)
                else:
                    utils.savePNG(path_png + '/' + imgDiffuseName,imgDiffuse, imgDiffuseNameFile,self.isName)
                    utils.savePNG(path_png + '/' + imgResidueName,imgResidue, imgResidueNameFile,self.isName)
                    utils.savePNG(path_png + '/' + imgSingelName,imgSingle, imgSingelNameFile,self.isName)
                    # utils.savePNG(path_png + '/' + imgSingelName_1,imgSingle_1, imgSingelNameFile_1,self.isName)
        
        path_npy = os.path.join(path_root_object, 'Show/npy_' + path_view)
        imgs_full_diffuse = np.array(imgs_full_diffuse,dtype=np.float32)
        imgs_full_residue = np.array(imgs_full_residue,dtype=np.float32)
        imgs_full_single = np.array(imgs_full_single,dtype=np.float32)
        img_full_diffuse = imgs_full_diffuse[3]
        img_full_residue = imgs_full_residue[3]
        img_full_single = imgs_full_single[3]
        np.save(path_npy + '/imgs_Full_diffuse_4whc.npy', imgs_full_diffuse)
        np.save(path_npy + '/imgs_Full_residue_4whc.npy', imgs_full_residue)
        np.save(path_npy + '/imgs_Full_single_4whc.npy', imgs_full_single)
        np.save(path_npy + '/img_Full_diffuse_whc.npy', img_full_diffuse)
        np.save(path_npy + '/img_Full_residue_whc.npy', img_full_residue)
        np.save(path_npy + '/img_Full_single_whc.npy', img_full_single)


        ## 分离前点照明图像
        file_frontLit_cross = path_root_view + '/Front_Cross_L' + fields[1].zfill(3) + '.png'
        file_frontLit_parallel = path_root_view + '/Front_Parallel_L' + fields[1].zfill(3) + '.png'
        srcCrossPath = file_frontLit_cross
        srcParallelPath = file_frontLit_parallel
        imgCross = utils.readPNG(srcCrossPath, self.resize, self.gamma)
        imgParallel = utils.readPNG(srcParallelPath, self.resize, self.gamma)
        imgDiffuse, imgResidue, _ = utils.splitSingleDiffuseSpecular(imgCross, imgParallel)
        # imgSingle_1 = imgDiffuse+ imgResidue
        imgSingle = imgDiffuse+ imgResidue
        path_npy = os.path.join(path_root_object, 'Show/npy_' + path_view)
        np.save(path_npy + '/img_Front_diffuse_whc.npy', imgDiffuse)
        np.save(path_npy + '/img_Front_residue_whc.npy', imgResidue)
        np.save(path_npy + '/img_Front_single_whc.npy', imgSingle)
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            imgDiffuseName = os.path.split(srcCrossPath)[1].replace('Cross', 'Diffuse')
            imgResidueName = os.path.split(srcParallelPath)[1].replace('Parallel', 'Residue')
            imgSingelName = os.path.split(srcCrossPath)[1].replace('Cross', 'Single')
            # imgSingelName_1 = os.path.split(srcCrossPath)[1].replace('Cross', 'Single_1')
            imgDiffuseNameFile = os.path.splitext(imgDiffuseName)[0]
            imgResidueNameFile = os.path.splitext(imgResidueName)[0]
            imgSingelNameFile = os.path.splitext(imgSingelName)[0]
            # imgSingelNameFile_1 = os.path.splitext(imgSingelName_1)[0]
            
            imgDiffuse = np.power(imgDiffuse, 1/self.gamma)
            imgResidue = np.power(imgResidue, 1/self.gamma)
            imgSingle = np.power(imgSingle, 1/self.gamma)
            # imgSingle_1 = np.power(imgSingle_1, 1/self.gamma)
            if self.isMax:
                utils.savePNG(path_png + '/' + imgDiffuseName,imgDiffuse/imgDiffuse.max(), imgDiffuseNameFile,self.isName)
                utils.savePNG(path_png + '/' + imgResidueName,imgResidue/imgResidue.max(), imgResidueNameFile,self.isName)
                utils.savePNG(path_png + '/' + imgSingelName,imgSingle/imgSingle.max(), imgSingelNameFile,self.isName)
                # utils.savePNG(path_png + '/' + imgSingelName_1,imgSingle_1/imgSingle_1.max(), imgSingelNameFile_1,self.isName)
            else:
                utils.savePNG(path_png + '/' + imgDiffuseName,imgDiffuse, imgDiffuseNameFile,self.isName)
                utils.savePNG(path_png + '/' + imgResidueName,imgResidue, imgResidueNameFile,self.isName)
                utils.savePNG(path_png + '/' + imgSingelName,imgSingle, imgSingelNameFile,self.isName)
                # utils.savePNG(path_png + '/' + imgSingelName_1,imgSingle_1, imgSingelNameFile_1,self.isName)
        
        
        
        
        end_X00 = time.time()
        print('X00 time is: ', end_X00 - start_X00)


    def X01_calculateNormalandMask(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X02 = time.time()

        path_npy = path_root_object + '/Show/npy_' + path_view
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37
        viewDirs = np.load(self.path_viewDirs)
        viewDir = viewDirs[camera_index, :]

        imgs_diffuse = np.load(path_npy +'/'+ self.path_imgs_Full_diffuse)
        imgs_residue = np.load(path_npy +'/'+ self.path_imgs_Full_residue)
        imgs_single = np.load(path_npy +'/'+ self.path_imgs_Full_single)
        
        img_d_c, img_s_c, img_d_s_c = utils.splitDiffuseSpecularGamma(imgs_diffuse[0], imgs_residue[0], imgs_single[0],gamma=1)
        img_d_x, img_s_x, img_d_s_x = utils.splitDiffuseSpecularGamma(imgs_diffuse[1], imgs_residue[1], imgs_single[1],gamma=1)
        img_d_y, img_s_y, img_d_s_y = utils.splitDiffuseSpecularGamma(imgs_diffuse[2], imgs_residue[2], imgs_single[2],gamma=1)
        img_d_z, img_s_z, img_d_s_z = utils.splitDiffuseSpecularGamma(imgs_diffuse[3], imgs_residue[3], imgs_single[3],gamma=1)
        
        img_d_c_gamma, img_s_c_gamma, img_d_s_c_gamma = utils.splitDiffuseSpecularGamma(imgs_diffuse[0], imgs_residue[0], imgs_single[0],self.gamma)
        img_d_x_gamma, img_s_x_gamma, img_d_s_x_gamma = utils.splitDiffuseSpecularGamma(imgs_diffuse[1], imgs_residue[1], imgs_single[1],self.gamma)
        img_d_y_gamma, img_s_y_gamma, img_d_s_y_gamma = utils.splitDiffuseSpecularGamma(imgs_diffuse[2], imgs_residue[2], imgs_single[2],self.gamma)
        img_d_z_gamma, img_s_z_gamma, img_d_s_z_gamma = utils.splitDiffuseSpecularGamma(imgs_diffuse[3], imgs_residue[3], imgs_single[3],self.gamma)

        ## 计算mask
        map_mask_gray = utils.threshold_mask(img_d_z, self.weight_mask)
        map_mask = cv2.merge([map_mask_gray,map_mask_gray,map_mask_gray])

        ## 未经gamma恢复的法线
        # 计算漫反射法线
        # b通道
        img_d_g_c = img_d_c[:,:,2]
        img_d_g_x = img_d_x[:,:,2]
        img_d_g_y = img_d_y[:,:,2]
        img_d_g_z = img_d_z[:,:,2]
        # # 灰度通道
        # img_d_g_c = (img_d_c[:,:,0]+img_d_c[:,:,1]+img_d_c[:,:,2]) /3
        # img_d_g_x = (img_d_x[:,:,0]+img_d_x[:,:,1]+img_d_x[:,:,2]) /3
        # img_d_g_y = (img_d_y[:,:,0]+img_d_y[:,:,1]+img_d_y[:,:,2]) /3
        # img_d_g_z = (img_d_z[:,:,0]+img_d_z[:,:,1]+img_d_z[:,:,2]) /3
        map_n_d_x = np.divide(2 * img_d_g_x - img_d_g_c, img_d_g_c, out=np.zeros_like(img_d_g_x), where=img_d_g_c != 0)
        map_n_d_y = np.divide(2 * img_d_g_y - img_d_g_c, img_d_g_c, out=np.zeros_like(img_d_g_y), where=img_d_g_c != 0)
        map_n_d_z = np.divide(2 * img_d_g_z - img_d_g_c, img_d_g_c, out=np.zeros_like(img_d_g_z), where=img_d_g_c != 0)

        map_n_d = cv2.merge([map_n_d_x, map_n_d_y, map_n_d_z])* map_mask
        map_n_d = map_n_d.clip(-1,1)
        # map_n_d = utils.normalImage(map_n_d)
        
        # 获得混合图像法线
        # 选择blue通道或者进行平均加权
        img_mix_c = img_d_s_c[:, :, 2]
        img_mix_x = img_d_s_x[:, :, 2]
        img_mix_y = img_d_s_y[:, :, 2]
        img_mix_z = img_d_s_z[:, :, 2]
        # img_mix_c = (img_ds_c[:, :, 0]+img_ds_c[:, :, 1]+img_ds_c[:, :, 2]) /3
        # img_mix_x = (img_ds_x[:, :, 0]+img_ds_x[:, :, 1]+img_ds_x[:, :, 2]) /3
        # img_mix_y = (img_ds_y[:, :, 0]+img_ds_y[:, :, 1]+img_ds_y[:, :, 2]) /3
        # img_mix_z = (img_ds_z[:, :, 0]+img_ds_z[:, :, 1]+img_ds_z[:, :, 2]) /3
        
        map_n_mix_x = np.divide(2 * img_mix_x - img_mix_c, img_mix_c, out=np.zeros_like(img_mix_x), where=img_mix_c != 0)
        map_n_mix_y = np.divide(2 * img_mix_y - img_mix_c, img_mix_c, out=np.zeros_like(img_mix_y), where=img_mix_c != 0)
        map_n_mix_z = np.divide(2 * img_mix_z - img_mix_c, img_mix_c, out=np.zeros_like(img_mix_z), where=img_mix_c != 0)

        map_n_mix = cv2.merge([map_n_mix_x, map_n_mix_y, map_n_mix_z]) *map_mask
        map_n_mix = map_n_mix.clip(-1,1)
        # map_n_mix = utils.normalImage(map_n_mix)
        
        # 获得反射法线
        img_r_g_c = img_s_c[:,:,2]
        img_r_g_x = img_s_x[:,:,2]
        img_r_g_y = img_s_y[:,:,2]
        img_r_g_z = img_s_z[:,:,2]
        # img_r_g_c = (img_s_c[:,:,0]+img_s_c[:,:,1]+img_s_c[:,:,2]) /3
        # img_r_g_x = (img_s_x[:,:,0]+img_s_x[:,:,1]+img_s_x[:,:,2]) /3
        # img_r_g_y = (img_s_y[:,:,0]+img_s_y[:,:,1]+img_s_y[:,:,2]) /3
        # img_r_g_z = (img_s_z[:,:,0]+img_s_z[:,:,1]+img_s_z[:,:,2]) /3
        map_n_r_x = np.divide(2 * img_r_g_x - img_r_g_c, img_r_g_c, out=np.zeros_like(img_r_g_x), where=img_r_g_c != 0)
        map_n_r_y = np.divide(2 * img_r_g_y - img_r_g_c, img_r_g_c, out=np.zeros_like(img_r_g_y), where=img_r_g_c != 0)
        map_n_r_z = np.divide(2 * img_r_g_z - img_r_g_c, img_r_g_c, out=np.zeros_like(img_r_g_z), where=img_r_g_c != 0)

        map_n_r = cv2.merge([map_n_r_x, map_n_r_y, map_n_r_z])* map_mask
        map_n_r = map_n_r.clip(-1,1)
        # map_n_r = utils.normalImage(map_n_r)
        
        map_view = utils.getViewDirImage(map_n_r, viewDir) *map_mask
        
        map_n_s = np.add(map_n_r, map_view,out=np.zeros_like(map_n_r),where=map_n_r!=0) /2* map_mask
        map_n_s = map_n_s.clip(-1,1)
        # map_n_s = utils.normalImage(map_n_s)
        
        
        
        ## 经gamma恢复的法线，用于高光贴图拟合
        # 计算漫反射法线
        # b通道
        img_d_g_c_gamma = img_d_c_gamma[:,:,2]
        img_d_g_x_gamma = img_d_x_gamma[:,:,2]
        img_d_g_y_gamma = img_d_y_gamma[:,:,2]
        img_d_g_z_gamma = img_d_z_gamma[:,:,2]
        # # 灰度通道
        # img_d_g_c_gamma = (img_d_c_gamma[:,:,0]+img_d_c_gamma[:,:,1]+img_d_c_gamma[:,:,2]) /3
        # img_d_g_x_gamma = (img_d_x_gamma[:,:,0]+img_d_x_gamma[:,:,1]+img_d_x_gamma[:,:,2]) /3
        # img_d_g_y_gamma = (img_d_y_gamma[:,:,0]+img_d_y_gamma[:,:,1]+img_d_y_gamma[:,:,2]) /3
        # img_d_g_z_gamma = (img_d_z_gamma[:,:,0]+img_d_z_gamma[:,:,1]+img_d_z_gamma[:,:,2]) /3
        map_n_d_x_gamma = np.divide(2 * img_d_g_x_gamma - img_d_g_c_gamma, img_d_g_c_gamma, out=np.zeros_like(img_d_g_x_gamma), where=img_d_g_c_gamma != 0)
        map_n_d_y_gamma = np.divide(2 * img_d_g_y_gamma - img_d_g_c_gamma, img_d_g_c_gamma, out=np.zeros_like(img_d_g_y_gamma), where=img_d_g_c_gamma != 0)
        map_n_d_z_gamma = np.divide(2 * img_d_g_z_gamma - img_d_g_c_gamma, img_d_g_c_gamma, out=np.zeros_like(img_d_g_z_gamma), where=img_d_g_c_gamma != 0)

        map_n_d_gamma = cv2.merge([map_n_d_x_gamma, map_n_d_y_gamma, map_n_d_z_gamma])* map_mask
        map_n_d_gamma = map_n_d_gamma.clip(-1,1)
        # map_n_d_gamma = utils.normalImage(map_n_d_gamma)
        
        # 获得混合图像法线
        # 选择blue通道或者进行平均加权
        img_mix_c_gamma = img_d_s_c_gamma[:, :, 2]
        img_mix_x_gamma = img_d_s_x_gamma[:, :, 2]
        img_mix_y_gamma = img_d_s_y_gamma[:, :, 2]
        img_mix_z_gamma = img_d_s_z_gamma[:, :, 2]
        # img_mix_c_gamma = (img_ds_c_gamma[:, :, 0]+img_ds_c_gamma[:, :, 1]+img_ds_c_gamma[:, :, 2]) /3
        # img_mix_x_gamma = (img_ds_x_gamma[:, :, 0]+img_ds_x_gamma[:, :, 1]+img_ds_x_gamma[:, :, 2]) /3
        # img_mix_y_gamma = (img_ds_y_gamma[:, :, 0]+img_ds_y_gamma[:, :, 1]+img_ds_y_gamma[:, :, 2]) /3
        # img_mix_z_gamma = (img_ds_z_gamma[:, :, 0]+img_ds_z_gamma[:, :, 1]+img_ds_z_gamma[:, :, 2]) /3
        
        map_n_mix_x_gamma = np.divide(2 * img_mix_x_gamma - img_mix_c_gamma, img_mix_c_gamma, out=np.zeros_like(img_mix_x_gamma), where=img_mix_c_gamma != 0)
        map_n_mix_y_gamma = np.divide(2 * img_mix_y_gamma - img_mix_c_gamma, img_mix_c_gamma, out=np.zeros_like(img_mix_y_gamma), where=img_mix_c_gamma != 0)
        map_n_mix_z_gamma = np.divide(2 * img_mix_z_gamma - img_mix_c_gamma, img_mix_c_gamma, out=np.zeros_like(img_mix_z_gamma), where=img_mix_c_gamma != 0)

        map_n_mix_gamma = cv2.merge([map_n_mix_x_gamma, map_n_mix_y_gamma, map_n_mix_z_gamma]) *map_mask
        map_n_mix_gamma = map_n_mix_gamma.clip(-1,1)
        # map_n_mix_gamma = utils.normalImage(map_n_mix_gamma)
        
        # 高光b通道
        img_r_g_c_gamma = img_s_c_gamma[:,:,2]
        img_r_g_x_gamma = img_s_x_gamma[:,:,2]
        img_r_g_y_gamma = img_s_y_gamma[:,:,2]
        img_r_g_z_gamma = img_s_z_gamma[:,:,2]
        # # 灰度通道
        # img_r_g_c_gamma = (img_s_c_gamma[:,:,0]+img_s_c_gamma[:,:,1]+img_s_c_gamma[:,:,2]) /3
        # img_r_g_x_gamma = (img_s_x_gamma[:,:,0]+img_s_x_gamma[:,:,1]+img_s_x_gamma[:,:,2]) /3
        # img_r_g_y_gamma = (img_s_y_gamma[:,:,0]+img_s_y_gamma[:,:,1]+img_s_y_gamma[:,:,2]) /3
        # img_r_g_z_gamma = (img_s_z_gamma[:,:,0]+img_s_z_gamma[:,:,1]+img_s_z_gamma[:,:,2]) /3
        map_n_r_x_gamma = np.divide(2 * img_r_g_x_gamma - img_r_g_c_gamma, img_r_g_c_gamma, out=np.zeros_like(img_r_g_x_gamma), where=img_r_g_c_gamma != 0)
        map_n_r_y_gamma = np.divide(2 * img_r_g_y_gamma - img_r_g_c_gamma, img_r_g_c_gamma, out=np.zeros_like(img_r_g_y_gamma), where=img_r_g_c_gamma != 0)
        map_n_r_z_gamma = np.divide(2 * img_r_g_z_gamma - img_r_g_c_gamma, img_r_g_c_gamma, out=np.zeros_like(img_r_g_z_gamma), where=img_r_g_c_gamma != 0)

        map_n_r_gamma = cv2.merge([map_n_r_x_gamma, map_n_r_y_gamma, map_n_r_z_gamma])* map_mask
        map_n_r_gamma = map_n_r_gamma.clip(-1,1)
        # map_n_r_gamma = utils.normalImage(map_n_r_gamma)
        
        map_n_s_gamma = np.add(map_n_r_gamma, map_view,out=np.zeros_like(map_n_r_gamma),where=map_n_r_gamma!=0) /2* map_mask
        map_n_s_gamma = map_n_s_gamma.clip(-1,1)
        # map_n_s_gamma = utils.normalImage(map_n_s_gamma)
        
        
    
        ## 保存法线,后期渲染需要使用法线
        np.save(path_npy + '/' + self.path_map_normal, map_n_d)      # 使用漫反射法线
        # np.save(path_npy + '/' + self.path_map_normal, map_n_mix)      # 使用混合法线
        # np.save(path_npy + '/' + self.path_map_normal, map_n_d_s_gamma)      # 使用高光-漫反射混合法线
        # np.save(path_npy + '/' + self.path_map_normal, map_n_d_s_2)      # 使用高光-漫反射混合法线
        
        np.save(path_npy + '/map_mask.npy', map_mask)
        
        # np.save(path_npy + '/map_normal_r.npy', map_n_r)
        # np.save(path_npy + '/map_normal_r_gamma.npy', map_n_r_gamma)
        np.save(path_npy + '/map_normal_s.npy', map_n_s)
        np.save(path_npy + '/map_normal_s_gamma.npy', map_n_s_gamma)
        np.save(path_npy + '/map_normal_d.npy', map_n_d)
        np.save(path_npy + '/map_normal_d_gamma.npy', map_n_d_gamma)
        # np.save(path_npy + '/map_normal_mix.npy', map_n_mix)
        # np.save(path_npy + '/map_normal_ds_2.npy', map_n_d_s_2)

        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            gamma = 1
            map_n_d_save = np.power(map_n_d, 1/gamma) *0.5+0.5
            map_n_d_save_gamma = np.power(map_n_d_gamma, 1/gamma) *0.5+0.5
            map_n_mix_save = np.power(map_n_mix, 1/gamma) *0.5+0.5
            map_n_mix_save_gamma = np.power(map_n_mix_gamma, 1/gamma) *0.5+0.5
            
            map_n_r_save = np.power(map_n_r, 1/gamma) *0.5+0.5
            map_n_r_save_gamma = np.power(map_n_r_gamma, 1/gamma) *0.5+0.5
            map_n_s_save = np.power(map_n_s, 1/gamma) *0.5+0.5
            map_n_s_save_gamma = np.power(map_n_s_gamma, 1/gamma) *0.5+0.5
            # map_n_d_s_2_save = np.power(map_n_d_s_2, 1/gamma) *0.5+0.5
            
            utils.savePNG(path_png + '/map/map_mask.png',map_mask, 'map_mask',self.isName)
            utils.savePNG(path_png + '/map/map_normal_r.png',map_n_r_save, 'map_normal_r',self.isName)
            utils.savePNG(path_png + '/map/map_normal_r_gamma.png',map_n_r_save_gamma, 'map_normal_r_gamma',self.isName)
            utils.savePNG(path_png + '/map/map_normal_s.png',map_n_s_save, 'map_normal_s',self.isName)
            utils.savePNG(path_png + '/map/map_normal_s_gamma.png',map_n_s_save_gamma, 'map_normal_s_gamma',self.isName)
            utils.savePNG(path_png + '/map/map_normal_diff.png',map_n_d_save, 'map_normal_diff',self.isName)
            utils.savePNG(path_png + '/map/map_normal_diff_gamma.png',map_n_d_save_gamma, 'map_normal_diff_gamma',self.isName)
            utils.savePNG(path_png + '/map/map_normal_mix.png',map_n_mix_save, 'map_normal_mix',self.isName)
            utils.savePNG(path_png + '/map/map_normal_mix_gamma.png',map_n_mix_save_gamma, 'map_normal_mix_gamma',self.isName)
            # utils.savePNG(path_png + '/map/map_normal_ds_2.png',map_n_d_s_2_save, 'map_normal_ds_2',self.isName)
            
        end_X02 = time.time()
        print('X02 time is: ', end_X02 - start_X02)


    def X01_calculateSpecularAlbedo(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X03 = time.time()

        path_npy = path_root_object + '/Show/npy_' + path_view
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37
        print('camera_index',camera_index)
        # front_index = int(fields[1])
        
        # 3.1 预处理, 参数图像
        img_front = np.load(path_npy + '/' + self.path_img_front_residue)
        img_full = np.load(path_npy + '/'  + self.path_img_full_residue)
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        # map_normal = np.load(path_npy + '/map_normal_ds_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_r.npy')
        # map_normal = np.load(path_npy + '/map_normal_r_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_s.npy')
        map_normal = np.load(path_npy + '/map_normal_s_gamma.npy')
        
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        # map_normal = np.load(path_npy + '/map_normal_diff.npy')
        # map_normal = np.load(path_npy + '/map_normal_mix.npy')
        # map_normal = np.load(path_npy + '/map_normal_ds_2.npy')
        map_mask = np.load(path_npy + '/'  + self.path_map_mask)
        viewDirs = np.load(self.path_viewDirs)

        # 归一化 img_front and img_full
        img_front = img_front / np.max(img_front)
        img_full = img_full / np.max(img_full)
        # map_normal = np.power(map_normal, self.gamma)
        
        result = {'DataTime': 0}

        # 3.2 根据角度以及亮度 #获得参与第一次拟合的像素区域
        # map_normal = map_normal / np.linalg.norm(map_normal,ord=2,axis=2).max()
        map_normal = map_normal / np.linalg.norm(map_normal,ord=2,axis=2,keepdims=True)
        viewDirs = viewDirs / np.linalg.norm(viewDirs, ord=2, axis=1, keepdims=True)
        viewDir = np.array(viewDirs[camera_index, :])  # 使用多个视点
        # print('viewDir_1.shape',viewDir_1.shape)
        # print('viewDir_1[0]', viewDir_1[0])

        max_angle = self.maxAngle
        dark_thr = self.darkThreshold
        img_angle, map_mask_fit = utils.get_anglemask(path_root_object, path_view, viewDir, max_angle, dark_thr, map_normal,
                                                      img_front, map_mask, self.isSave)

        # 3.3 进行第一次拟合  假设逐像素 speclar albedo 都是1
        self.Fit_Counter = 1
        map_specular_1 = np.ones_like(img_full)
        result['Fit_First'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular_1, self.Fit_Counter)

        print('result_first', result['Fit_First'])
        # 3.4 :计算C （speclar albedo）
        map_specular = utils.calc_speclar(path_root_object, path_view, img_full, img_angle, map_mask, result, self.isSave)
        # map_specular = map_specular /2
        
        print('高光贴图的最大值', np.max(map_specular))
        print('高光贴图的平均值', np.mean(map_specular))
        print('高光贴图的最小值', np.min(map_specular))
        map_specular = utils.normalClipAndMax(map_specular, flag=2)   # flag=0:clip and max;flag=1:clip;flag=2:max
        print('\t')
        # 3.5:第2次拟合 p(h)的参数  C=img_speclar
        self.Fit_Counter = 2
        # result['Fit_Second'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)
        result['Fit_First'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)

        print('result_second', result['Fit_First'])
        map_specular = utils.calc_speclar(path_root_object, path_view, img_full, img_angle, map_mask, result, self.isSave)
        print('高光贴图的最大值', np.max(map_specular))
        print('高光贴图的平均值', np.mean(map_specular))
        print('高光贴图的最小值', np.min(map_specular))
        map_specular = utils.normalClipAndMax(map_specular, flag=2)   # flag=0:clip and max;flag=1:clip;flag=2:max
        print('\t')
        
        # # 3.5:第3次拟合 p(h)的参数  C=img_speclar
        # self.Fit_Counter = 3
        # # result['Fit_Second'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)
        # result['Fit_First'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)

        # print('result_third', result['Fit_First'])
        # map_specular = utils.calc_speclar(path_root_object, path_view, img_full, img_angle, map_mask, result, self.isSave)
        # print('高光贴图的最大值', np.max(map_specular))
        # print('高光贴图的平均值', np.mean(map_specular))
        # print('高光贴图的最小值', np.min(map_specular))
        # map_specular = utils.normalClipAndMax(map_specular, flag=2)   # flag=0:clip and max;flag=1:clip;flag=2:max
        # print('\t')
        
        # # 3.5:第4次拟合 p(h)的参数  C=img_speclar
        # self.Fit_Counter = 4
        # # result['Fit_Second'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)
        # result['Fit_First'] = utils.fit_region(self.r0, map_mask_fit, img_angle, img_front, map_specular, self.Fit_Counter)

        # print('result_third', result['Fit_First'])
        # map_specular = utils.calc_speclar(path_root_object, path_view, img_full, img_angle, map_mask, result, self.isSave)
        # print('高光贴图的最大值', np.max(map_specular))
        # print('高光贴图的平均值', np.mean(map_specular))
        # print('高光贴图的最小值', np.min(map_specular))
        # map_specular = utils.normalClipAndMax(map_specular, flag=2)   # flag=0:clip and max;flag=1:clip;flag=2:max
        # print('\t')
        
        ## 渲染验证
        # map_normal = np.load(path_npy + '/map_normal_diff.npy')
        # map_normal = np.load(path_npy + '/map_normal_spec.npy')
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        
        # map_normal = np.load(path_npy + '/map_normal_mix.npy')
        # map_normal = np.load(path_npy + '/map_normal_spec_2.npy')
        # map_normal = np.load(path_npy + '/map_normal_ds_2.npy')
        
        map_specular = np.add(map_specular,img_full,out=2*img_full,where=map_specular!=0)/2 * map_mask
        
        ## 保存数据
        np.save(path_npy + '/' + self.path_map_specular, map_specular)  # [h,w,c]
        np.save(path_npy + '/' + self.path_result, result)
        np.save(path_npy + '/' + self.path_viewDirs, viewDirs)
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            map_specular = np.power(map_specular, 1/self.gamma)
            if self.isMax:
                utils.savePNG(path_png + '/map/map_sscatter.png',map_specular/map_specular.max(), 'map_specular',self.isName)
            else:
                utils.savePNG(path_png + '/map/map_specular.png',map_specular, 'map_specular',self.isName)
            
        end_X03 = time.time()
        print('X03 time is: ', end_X03 - start_X03)


    def X01_calculateSScatterAlbedo(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X03 = time.time()

        path_npy = path_root_object + '/Show/npy_' + path_view
        # fields = path_view.split('_')
        # camera_index = int(fields[0].replace('Material', '')) - 37

        img_front = np.load(path_npy + '/' + self.path_img_front_residue)
        img_full = np.load(path_npy + '/'  + self.path_img_full_residue)
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        map_normal = np.load(path_npy + '/map_normal_d_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_r.npy')
        # map_normal = np.load(path_npy + '/map_normal_r_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_s.npy')
        # map_normal = np.load(path_npy + '/map_normal_s_gamma.npy')
        map_mask = np.load(path_npy + '/'  + self.path_map_mask)
        map_specular = np.load(path_npy + '/' + self.path_map_specular)  # [h,w,c]
        
        
        # 归一化 img_front and img_full
        img_front = img_front / np.max(img_front)
        img_full = img_full / np.max(img_full)
        
        map_sscatter = (img_full - map_specular * self.temp_ss) * map_mask
        print('单次散射albedo的最大值', np.max(map_sscatter))
        print('单次散射albedo的平均值', np.mean(map_sscatter))
        print('单次散射albedo的最小值', np.min(map_sscatter))
        map_sscatter = utils.normalClipAndMax(map_sscatter, flag=2)   # flag=0:clip and max;flag=1:clip;flag=2:max
        print('单次散射albedo的最大值', np.max(map_sscatter))
        print('单次散射albedo的平均值', np.mean(map_sscatter))
        print('\t')
        
        ## 保存数据
        np.save(path_npy + '/' + self.path_map_sscatter, map_sscatter)  # [h,w,c]
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            map_sscatter = np.power(map_sscatter, 1/self.gamma)
            # print('map_sscatter.shape',map_sscatter.shape)
            if self.isMax:
                utils.savePNG(path_png + '/map/map_sscatter.png',map_sscatter/map_sscatter.max(), 'map_sscatter',self.isName)
                
            else:
                utils.savePNG(path_png + '/map/map_sscatter.png',map_sscatter, 'map_sscatter',self.isName)
            
        end_X03 = time.time()
        print('X03 time is: ', end_X03 - start_X03)


    def X01_calculateDiffuseAlbedo(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X04 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view
        # fields = path_view.split('_')
        # camera_index = int(fields[0].replace('Material', '')) - 37
        
        map_mask = np.load(path_npy + '/'  + self.path_map_mask)
        full_diffuse = np.load(path_npy + '/' + self.path_img_full_diffuse)
        # front_diffuse = np.load(path_npy + '/' + self.path_img_front_diffuse)
        # result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        map_diffuse = full_diffuse * map_mask
        map_diffuse = utils.normalClipAndMax(map_diffuse, flag=3)   # flag=0:clip and max;flag=1:clip;flag=2:max
        np.save(path_npy + '/' + self.path_map_diffuse, map_diffuse)
        
        # ## 渲染单光照漫反射
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        # viewDirs = np.load(path_npy + '/' + self.path_viewDirs)
        # viewDir = np.array(viewDirs[camera_index, :])  # 使用多个视点
        # viewDir = viewDir / np.linalg.norm(viewDir)
        
        # render_shading = utils.singleShadingRender(map_normal, map_mask, viewDir)
        # render_frontlit_d = utils.singleDiffuseRender(map_diffuse,render_shading,self.ambientIntensity) * map_mask
        # print('render_frontlit_d.max()', render_frontlit_d.max())
        # print('render_frontlit_d.mean()', render_frontlit_d.mean())
        # print('render_frontlit_d.min()', render_frontlit_d.min())
        
        # ## 计算白平衡
        # front_diffuse = front_diffuse * map_mask
        # wb_r = (front_diffuse[:,:,0]).mean() / (render_frontlit_d[:,:,0]).mean()
        # wb_g = (front_diffuse[:,:,1]).mean() / (render_frontlit_d[:,:,1]).mean()
        # wb_b = (front_diffuse[:,:,2]).mean() / (render_frontlit_d[:,:,2]).mean()
        # val_r=render_frontlit_d[:,:,0]*wb_r
        # val_g=render_frontlit_d[:,:,1]*wb_g
        # val_b=render_frontlit_d[:,:,2]*wb_b
        # render_frontlit_d=(cv2.merge([val_r,val_g,val_b])).astype('float32')
        # print('wb',wb_r,wb_g,wb_b)
        # print('result_first', result['Fit_First'])
        # print('render_frontlit_d.max()', render_frontlit_d.max())
        # print('render_frontlit_d.mean()', render_frontlit_d.mean())
        # print('render_frontlit_d.min()', render_frontlit_d.min())
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            map_diffuse = np.power(map_diffuse, 1/self.gamma)
            # render_frontlit_d = np.power(render_frontlit_d, 1/self.gamma)
            if self.isMax:
                utils.savePNG(path_png + '/map/map_diffuse.png',map_diffuse/map_diffuse.max(), 'map_diffuse',self.isName)
                # utils.savePNG(path_png + '/render_frontlit_d.png',render_frontlit_d/render_frontlit_d.max(), 'render_frontlit_d',self.isName)
            else:
                utils.savePNG(path_png + '/map/map_diffuse.png',map_diffuse, 'map_diffuse',self.isName)
                # utils.savePNG(path_png + '/render_frontlit_d.png',render_frontlit_d, 'render_frontlit_d',self.isName)

        print('漫反射albedo的最大值', np.max(map_diffuse))
        print('漫反射albedo的平均值', np.mean(map_diffuse))
            
        end_X04 = time.time()
        print('X04 time is: ', end_X04 - start_X04)


    def X01_renderFrontlitImage(self,path_root_object, path_view):
        start_X03 = time.time()

        path_npy = path_root_object + '/Show/npy_' + path_view
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37

        front_residue = np.load(path_npy + '/' + self.path_img_front_residue)
        # full_residue = np.load(path_npy + '/'  + self.path_img_full_residue)
        front_diffuse = np.load(path_npy + '/' + self.path_img_front_diffuse)
        # map_normal = np.load(path_npy + '/'  + self.path_map_normal)
        # map_normal = np.load(path_npy + '/map_normal_ds.npy')
        map_normal = np.load(path_npy + '/map_normal_d_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_r.npy')
        # map_normal = np.load(path_npy + '/map_normal_r_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_s.npy')
        # map_normal = np.load(path_npy + '/map_normal_s_gamma.npy')
        
        map_mask = np.load(path_npy + '/'  + self.path_map_mask)
        map_specular = np.load(path_npy + '/' + self.path_map_specular)  # [h,w,c]
        map_sscatter = np.load(path_npy + '/' + self.path_map_sscatter)  # [h,w,c]
        map_diffuse = np.load(path_npy + '/' + self.path_map_diffuse)
        result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        
        viewDirs = np.load(path_npy + '/' + self.path_viewDirs)
        
        k2 = viewDirs[camera_index, 0:]
        k2 = k2 / np.linalg.norm(k2, ord=2)
        k1 = k2
        
        print('result_first', result['Fit_First'])
        alpha = result['Fit_First']['alpha']
        intensity = result['Fit_First']['intensity']
        current_flag = 1
        
        # Step 0: render transmittance, inclinding dt_L and dt_V
        # 如果不存在预制的rho_dt,则重新计算
        if os.path.exists(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy'):
            rho_dt = np.load(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy')
        else:
            ## 逐个计算,预存rho_dt
            costheta = np.linspace(0, 1, num=self.rho_dt_num)
            # print('costheta',costheta)
            rho_dt = np.zeros_like(costheta)
            for i in range(len(costheta)):
                rho_dt[i] = utils.computeRhodt(costheta[i], self.r0, alpha, intensity, numterms=80)
            np.save(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy', rho_dt)
        
        k2 = viewDirs[camera_index, 0:]
        k2 = k2 / np.linalg.norm(k2, ord=2)
        k1 = k2
        
        render_dt_V = utils.computeRhodtMatrix(k2, map_normal, rho_dt) * map_mask
        render_dt_L = render_dt_V
        
        # Step 1: render shading component
        render_frontlit_sh = utils.singleShadingRender(map_normal, map_mask, k2,self.diffuseIntensity,self.weight_shading)
        render_frontlit_sh = utils.normalClipAndMax(render_frontlit_sh,current_flag)
        print('render_frontlit_sh.max()', render_frontlit_sh.max())
        print('render_frontlit_sh.mean()', render_frontlit_sh.mean())
        print('render_frontlit_sh.min()', render_frontlit_sh.min())
        print('\t')
        
        # Step 2: render single scatter component
        render_frontlit_ss = utils.singleSScatterRender(map_mask,map_normal,map_sscatter,k1,k2,render_dt_L,render_dt_V,render_frontlit_sh,self.weight_lambert,self.weight_shading) *self.k_ss
        render_frontlit_ss = utils.normalClipAndMax(render_frontlit_ss,current_flag)
        print('render_frontlit_ss.max()', render_frontlit_ss.max())
        print('render_frontlit_ss.mean()', render_frontlit_ss.mean())
        print('render_frontlit_ss.min()', render_frontlit_ss.min())
        print('\t')
        
        # Step 3: render specualr component
        render_frontlit_sp = utils.singleSpecularRender(k2,k2,map_normal,self.r0,alpha,intensity,map_specular[:,:,0],self.lightIntensity,self.weight_shading)
        render_frontlit_sp = render_frontlit_sp / render_frontlit_sp.mean() * render_frontlit_ss.mean() 
        render_frontlit_sp = utils.normalClipAndMax(render_frontlit_sp,current_flag)
        print('render_frontlit_sp.max()', render_frontlit_sp.max())
        print('render_frontlit_sp.mean()', render_frontlit_sp.mean())
        print('render_frontlit_sp.min()', render_frontlit_sp.min())
        print('\t')
        
        # Step 3_0: render residue component
        render_frontlit_rs = render_frontlit_sp + render_frontlit_ss
        render_frontlit_rs = utils.normalClipAndMax(render_frontlit_rs,current_flag)
        print('render_frontlit_rs.max()', render_frontlit_rs.max())
        print('render_frontlit_rs.mean()', render_frontlit_rs.mean())
        print('render_frontlit_rs.min()', render_frontlit_rs.min())
        print('\t')
        
        # front_residue = front_residue * map_mask
        # wb_rs = (front_residue[:,:,0]).mean() / (render_frontlit_rs[:,:,0]).mean()
        # # wb_rs = (front_residue[:,:,0]).max() / (render_frontlit_rs[:,:,0]).max()
        # render_frontlit_rs=render_frontlit_rs*wb_rs
        # render_frontlit_rs = utils.normalClipAndMax(render_frontlit_rs,current_flag)
        # print('render_frontlit_rs.max()', render_frontlit_rs.max())
        # print('render_frontlit_rs.mean()', render_frontlit_rs.mean())
        # print('render_frontlit_rs.min()', render_frontlit_rs.min())
        # result['Fit_First']['wb_rs'] = wb_rs
        # print('result_first', result['Fit_First'])
        # print('\t')
        
        # Step 4: render mix_irradiance component
        if self.mix == 0.5:
            map_diffuse_front = np.sqrt(map_diffuse)
            map_diffuse_back = map_diffuse_front
        else:
            map_diffuse_front = np.power(map_diffuse, self.mix)
            map_diffuse_back = np.power(map_diffuse, 1 - self.mix)
        
        irradiance = utils.computeIrradiance(k1, map_normal, map_specular, render_dt_L, map_mask,self.weight_shading)
        render_frontlit_irr = utils.gaussianBlur_sum(irradiance * map_diffuse_front,
                                                             self.gauss_kernel_size, self.sigma_magnification)
        render_frontlit_irr = utils.normalClipAndMax(render_frontlit_irr,current_flag)
        print('render_frontlit_irr.max()', render_frontlit_irr.max())
        print('render_frontlit_irr.mean()', render_frontlit_irr.mean())
        print('render_frontlit_irr.min()', render_frontlit_irr.min())
        print('\t')
        
        # Step 5: render subscatter
        render_frontlit_sub = utils.singleSubScatterRender(k2,map_diffuse_back,map_diffuse,map_specular,map_normal,map_mask,rho_dt,render_frontlit_irr,self.ambientIntensity)
        render_frontlit_sub = utils.normalClipAndMax(render_frontlit_sub,current_flag)
        print('render_frontlit_sub.max()', render_frontlit_sub.max())
        print('render_frontlit_sub.mean()', render_frontlit_sub.mean())
        print('render_frontlit_sub.min()', render_frontlit_sub.min())
        print('\t')
        
        # Step 6: calculate white balance values
        front_diffuse = front_diffuse * map_mask
        wb_r = (front_diffuse[:,:,0]).mean() / (render_frontlit_sub[:,:,0]).mean()
        wb_g = (front_diffuse[:,:,1]).mean() / (render_frontlit_sub[:,:,1]).mean()
        wb_b = (front_diffuse[:,:,2]).mean() / (render_frontlit_sub[:,:,2]).mean()
        val_r=render_frontlit_sub[:,:,0]*wb_r
        val_g=render_frontlit_sub[:,:,1]*wb_g
        val_b=render_frontlit_sub[:,:,2]*wb_b
        render_frontlit_sub=(cv2.merge([val_r,val_g,val_b])).astype('float32')
        render_frontlit_sub = utils.normalClipAndMax(render_frontlit_sub,current_flag)
        # print('wb',wb_r,wb_g,wb_b)
        print('render_frontlit_sub.max()', render_frontlit_sub.max())
        print('render_frontlit_sub.mean()', render_frontlit_sub.mean())
        print('render_frontlit_sub.min()', render_frontlit_sub.min())
        result['Fit_First']['wb_r'] = wb_r
        result['Fit_First']['wb_g'] = wb_g
        result['Fit_First']['wb_b'] = wb_b
        print('result_first', result['Fit_First'])
        print('\t')
        # Step 7: render diffuse
        render_frontlit_d = utils.singleDiffuseRender(map_diffuse,render_frontlit_sh,self.ambientIntensity) * map_mask
        render_frontlit_d = utils.normalClipAndMax(render_frontlit_d,current_flag)
        print('render_frontlit_d.max()', render_frontlit_d.max())
        print('render_frontlit_d.mean()', render_frontlit_d.mean())
        print('render_frontlit_d.min()', render_frontlit_d.min())
        print('\t')
        
        # front_diffuse = front_diffuse * map_mask
        # d_wb_r = (front_diffuse[:,:,0]).mean() / (render_frontlit_d[:,:,0]).mean()
        # d_wb_g = (front_diffuse[:,:,1]).mean() / (render_frontlit_d[:,:,1]).mean()
        # d_wb_b = (front_diffuse[:,:,2]).mean() / (render_frontlit_d[:,:,2]).mean()
        # d_val_r=render_frontlit_d[:,:,0]*d_wb_r
        # d_val_g=render_frontlit_d[:,:,1]*d_wb_g
        # d_val_b=render_frontlit_d[:,:,2]*d_wb_b
        # render_frontlit_d=(cv2.merge([d_val_r,d_val_g,d_val_b])).astype('float32')
        # print('wb',d_wb_r,d_wb_g,d_wb_b)
        # print('render_frontlit_d.max()', render_frontlit_d.max())
        # print('render_frontlit_d.mean()', render_frontlit_d.mean())
        # print('render_frontlit_d.min()', render_frontlit_d.min())
        
        sub_val_r=render_frontlit_d[:,:,0]*wb_r
        sub_val_g=render_frontlit_d[:,:,1]*wb_g
        sub_val_b=render_frontlit_d[:,:,2]*wb_b
        render_frontlit_sub_d=(cv2.merge([sub_val_r,sub_val_g,sub_val_b])).astype('float32')
        
        # Step 8: merge render image
        # render_frontlit_our_0 = np.add(render_frontlit_sp,render_frontlit_ss*self.k_ss,out=render_frontlit_ss,where=render_frontlit_sp!=0)
        render_frontlit_our = render_frontlit_rs +render_frontlit_sub
        render_frontlit_our = utils.normalClipAndMax(render_frontlit_our,1)
        print('render_frontlit_our.max()', render_frontlit_our.max())
        print('render_frontlit_our.mean()', render_frontlit_our.mean())
        print('render_frontlit_our.min()', render_frontlit_our.min())
        print('\t')
        # render_frontlit_other = render_frontlit_sp +render_frontlit_d
        render_frontlit_other = render_frontlit_sp +render_frontlit_sub_d
        render_frontlit_other = utils.normalClipAndMax(render_frontlit_other,1)
        print('render_frontlit_other.max()', render_frontlit_other.max())
        print('render_frontlit_other.mean()', render_frontlit_other.mean())
        print('render_frontlit_other.min()', render_frontlit_other.min())
        print('\t')
        ## 保存数据
        np.save(path_npy + '/' + self.path_result, result)
        if self.isSave:
        # if True:
            path_png = path_root_object + '/Show/image_' + path_view
            gamma = self.gamma
            # gamma = 1
            render_frontlit_sp = np.power(render_frontlit_sp, 1/gamma)
            render_frontlit_sh = np.power(render_frontlit_sh, 1/gamma)
            render_frontlit_ss = np.power(render_frontlit_ss, 1/gamma)
            render_frontlit_rs = np.power(render_frontlit_rs, 1/gamma)
            render_frontlit_irr = np.power(render_frontlit_irr, 1/gamma)
            render_frontlit_sub = np.power(render_frontlit_sub, 1/gamma)
            render_frontlit_d = np.power(render_frontlit_d, 1/gamma)
            render_frontlit_sub_d = np.power(render_frontlit_sub_d, 1/gamma)
            render_frontlit_our = np.power(render_frontlit_our, 1/gamma)
            render_frontlit_other = np.power(render_frontlit_other, 1/gamma)
            if self.isMax:
                utils.savePNG(path_png + '/render_frontlit_sp.png',render_frontlit_sp/render_frontlit_sp.max(), 'render_frontlit_sp',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sh.png',render_frontlit_sh/render_frontlit_sh.max(), 'render_frontlit_sh',self.isName)
                utils.savePNG(path_png + '/render_frontlit_ss.png',render_frontlit_ss/render_frontlit_ss.max(), 'render_frontlit_ss',self.isName)
                utils.savePNG(path_png + '/render_frontlit_rs.png',render_frontlit_rs/render_frontlit_rs.max(), 'render_frontlit_rs',self.isName)
                utils.savePNG(path_png + '/render_frontlit_irr.png',render_frontlit_irr/render_frontlit_irr.max(), 'render_frontlit_irr',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sub.png',render_frontlit_sub/render_frontlit_sub.max(), 'render_frontlit_sub',self.isName)
                utils.savePNG(path_png + '/render_frontlit_d.png',render_frontlit_d/render_frontlit_d.max(), 'render_frontlit_d',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sub_d.png',render_frontlit_sub_d/render_frontlit_sub_d.max(), 'render_frontlit_sub_d',self.isName)
                utils.savePNG(path_png + '/render_frontlit_our.png',render_frontlit_our/render_frontlit_our.max(), 'render_frontlit_our',self.isName)
                utils.savePNG(path_png + '/render_frontlit_other.png',render_frontlit_other/render_frontlit_other.max(), 'render_frontlit_other',self.isName)
                
            else:
                utils.savePNG(path_png + '/render_frontlit_sp.png',render_frontlit_sp, 'render_frontlit_sp',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sh.png',render_frontlit_sh, 'render_frontlit_sh',self.isName)
                utils.savePNG(path_png + '/render_frontlit_ss.png',render_frontlit_ss, 'render_frontlit_ss',self.isName)
                utils.savePNG(path_png + '/render_frontlit_rs.png',render_frontlit_rs, 'render_frontlit_rs',self.isName)
                utils.savePNG(path_png + '/render_frontlit_irr.png',render_frontlit_irr, 'render_frontlit_irr',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sub.png',render_frontlit_sub, 'render_frontlit_sub',self.isName)
                utils.savePNG(path_png + '/render_frontlit_d.png',render_frontlit_d, 'render_frontlit_d',self.isName)
                utils.savePNG(path_png + '/render_frontlit_sub_d.png',render_frontlit_sub_d, 'render_frontlit_sub_d',self.isName)
                utils.savePNG(path_png + '/render_frontlit_our.png',render_frontlit_our, 'render_frontlit_our',self.isName)
                utils.savePNG(path_png + '/render_frontlit_other.png',render_frontlit_other, 'render_frontlit_other',self.isName)
                
            
        end_X03 = time.time()
        print('X03 time is: ', end_X03 - start_X03)
        
    
    def X01_renderTransmittance(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X06 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view
        
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37
        # front_index = int(fields[1])
        viewDirs = np.load(path_npy + '/' + self.path_viewDirs)

        map_normal = np.load(path_npy + '/' + self.path_map_normal)
        map_mask = np.load(path_npy + '/' + self.path_map_mask)
        
        # lightDirs = np.load(self.path_lightDirs)
        lightDirs = np.load(self.path_lightDirsFront)
        result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        # height, width, channel = map_normal.shape
        num_lights = lightDirs.shape[0]

        ## 计算依赖于ndotL的rho_dt,size is [rho_dt_num,1]
        print('result_first', result['Fit_First'])
        alpha = result['Fit_First']['alpha']
        intensity = result['Fit_First']['intensity']
        # s1 = result['Fit_First']['shininess1']
        # s2 = result['Fit_First']['shininess1']
        # ks1 = result['Fit_First']['ks1']
        # ks2 = result['Fit_First']['ks2']
        # 如果不存在预制的rho_dt,则重新计算
        if os.path.exists(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy'):
            rho_dt = np.load(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy')
        else:
            ## 逐个计算,预存rho_dt
            costheta = np.linspace(0, 1, num=self.rho_dt_num)
            # print('costheta',costheta)
            rho_dt = np.zeros_like(costheta)
            for i in range(len(costheta)):
                rho_dt[i] = utils.computeRhodt(costheta[i], self.r0, alpha, intensity, numterms=80)
            np.save(path_npy + '/rho_dt' + f'_{self.rho_dt_num}.npy', rho_dt)
        
        ## 计算依赖于ndotV的rho_dt矩阵
        k2 = viewDirs[camera_index, 0:]
        k2 = k2 / np.linalg.norm(k2, ord=2)
        rho_dt_V = utils.computeRhodtMatrix(k2, map_normal, rho_dt) * map_mask

        # np.save(path_npy + '/' + self.path_lightDirs, lightDirs)
        # np.save(path_npy + '/' + self.path_lightDirsFront, lightDirs)
        np.save(path_npy + '/'+self.path_render_rho_dt_V, rho_dt_V)
        
        ## 计算依赖于ndotL的rho_dt矩阵, size is [num_lights,h,w,c]
        render_rho_dt_Ls = []
        for k in range(num_lights):
            k1 = lightDirs[k, 0:]
            k1 = k1 / np.linalg.norm(k1, ord=2)
            rho_dt_L_matric_i = utils.computeRhodtMatrix(k1, map_normal, rho_dt) * map_mask
            # print('rho_dt_L_matric_i.max()',rho_dt_L_matric_i.max())
            # print('rho_dt_L_matric_i.min()',rho_dt_L_matric_i.min())
            render_rho_dt_Ls.append(rho_dt_L_matric_i)
            
            # if self.isSave:
            #     path_png = path_root_object + '/Show/image_' + path_view
            #     # gamma = self.gamma
            #     # rho_dt_L_matric_i = np.power(rho_dt_L_matric_i, 1/gamma)
            #     if self.isMax:
            #         utils.savePNG(path_png + '/render_rho_dt/' + f'{str(k).zfill(4)}.png',rho_dt_L_matric_i/rho_dt_L_matric_i.max(), 'rho_dt_L_matric_i',self.isName)
            #     else:
            #         utils.savePNG(path_png + '/render_rho_dt/' + f'{str(k).zfill(4)}.png',rho_dt_L_matric_i, 'rho_dt_L_matric_i',self.isName)
                
        # 保存渲染结果
        render_rho_dt_Ls = np.array(render_rho_dt_Ls).astype('float32')
        np.save(path_npy + '/'+self.path_render_rho_dt_Ls, render_rho_dt_Ls)
        

        end_X06 = time.time()
        print('X06 time is: ', end_X06 - start_X06)
        

    def X02_renderShade(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X05 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view
        lightDirs = np.load(self.path_lightDirsFront)
        # lightDirs = np.load(self.path_lightDirs)
        # lightDirs = np.load(self.path_lightDirsFront)
        # map_normal = np.load(path_npy + '/' + self.path_map_normal)
        map_normal = np.load(path_npy + '/map_normal_d_gamma.npy')
        # map_normal = np.load(path_npy + '/map_normal_mix.npy')
        # map_normal = np.load(path_npy + '/map_normal_diff.npy')
        # map_normal = np.load(path_npy + '/map_normal_ds_2.npy')
        map_mask = np.load(path_npy + '/' + self.path_map_mask)
        render_unlit_shading = utils.unlitShadingRender(map_normal, map_mask, lightDirs,self.diffuseIntensity,self.weight_shading)  # [num_lights,h,w,c]
        print('阴影的最大值', np.max(render_unlit_shading))
        print('阴影的平均值', np.mean(render_unlit_shading))
        ## 保存无光阴影渲染结果
        np.save(path_npy + '/' + self.path_render_unlit_shading, render_unlit_shading)
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            for k in range(render_unlit_shading.shape[0]):
                render_shading = render_unlit_shading[k, :, :, :]
                # render_shading = np.power(render_shading, 1/self.gamma)
                if self.isMax:
                    utils.savePNG(path_png + '/render_shading/' + f'{str(k).zfill(4)}.png',render_shading/render_shading.max(), 'render_shading',self.isName)
                else:
                    utils.savePNG(path_png + '/render_shading/' + f'{str(k).zfill(4)}.png',render_shading, 'render_shading',self.isName)

        end_X05 = time.time()
        print('X05 time is: ', end_X05 - start_X05)


    def X02_renderMixIrradiance(self, path_root_object, path_view):
        start_X07 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view

        map_diffuse = np.load(path_npy + '/' + self.path_map_diffuse)
        map_normal = np.load(path_npy + '/' + self.path_map_normal)
        map_mask = np.load(path_npy + '/' + self.path_map_mask)
        map_specular = np.load(path_npy + '/' + self.path_map_specular)
        # lightDirs = np.load(self.path_lightDirs)
        lightDirs = np.load(self.path_lightDirsFront)
        render_rho_dt_Ls = np.load(path_npy + '/'+self.path_render_rho_dt_Ls)
        
        if self.mix == 0.5:
            map_diffuse_front = np.sqrt(map_diffuse)
            map_diffuse_back = map_diffuse_front
        else:
            map_diffuse_front = np.power(map_diffuse, self.mix)
            map_diffuse_back = np.power(map_diffuse, 1 - self.mix)
        np.save(path_npy + '/map_diffuse_back.npy', map_diffuse_back)
        
        render_unlit_mix_irradiance = []
        for k in range(lightDirs.shape[0]):
            k1 = lightDirs[k, 0:]
            k1 = k1 / np.linalg.norm(k1, ord=2)
            rho_dt_L_matric_i = render_rho_dt_Ls[k]
            irradiance = utils.computeIrradiance(k1, map_normal, map_specular, rho_dt_L_matric_i, map_mask,self.weight_shading)
            
            render_irradiance_front = utils.gaussianBlur_sum(irradiance * map_diffuse_front,
                                                             self.gauss_kernel_size, self.sigma_magnification)
            render_unlit_mix_irradiance.append(render_irradiance_front)
            
        ## 保存渲染数据
        render_unlit_mix_irradiance = np.array(render_unlit_mix_irradiance).astype('float32')
        np.save(path_npy + '/'+self.path_render_unlit_mix_irradiance, render_unlit_mix_irradiance)

        end_X07 = time.time()
        print('X07 time is: ', end_X07 - start_X07)


    def X02_renderSScatterComponent(self, path_root_object, path_view):
        start_X09 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view

        # 1st order single scattering BRDF model
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37
        # front_index = int(fields[1])
        # lightDirs = np.load(self.path_lightDirs)
        lightDirs = np.load(self.path_lightDirsFront)
        viewDirs = np.load(path_npy + '/' + self.path_viewDirs)
        map_sscatter = np.load(path_npy + '/' + self.path_map_sscatter)
        map_normal = np.load(path_npy + '/' + self.path_map_normal)
        map_mask = np.load(path_npy + '/' + self.path_map_mask)
        render_rho_dt_Ls = np.load(path_npy + '/'+self.path_render_rho_dt_Ls)
        rho_dt_V = np.load(path_npy + '/'+self.path_render_rho_dt_V)
        render_unlit_shading = np.load(path_npy + '/' + self.path_render_unlit_shading)
        result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        print('result_first', result['Fit_First'])
        viewDir = viewDirs[camera_index]
        ## 渲染单次散射分量
        render_unlit_sscatter = utils.unlitSScatterRender(lightDirs,viewDir,map_mask,map_normal,map_sscatter,render_rho_dt_Ls,rho_dt_V,render_unlit_shading,self.weight_lambert,self.weight_shading) *self.k_ss
        
        # wb_rs = result['Fit_First']['wb_rs']
        # render_unlit_sscatter = render_unlit_sscatter *wb_rs
        print('单次散射分量的最大值', np.max(render_unlit_sscatter))
        print('单次散射分量的平均值', np.mean(render_unlit_sscatter))
        render_unlit_sscatter = utils.normalClipAndMax(render_unlit_sscatter, flag=self.flag)
        print('单次散射分量的最大值', np.max(render_unlit_sscatter))
        print('单次散射分量的平均值', np.mean(render_unlit_sscatter))

        # 保存渲染结果
        np.save(path_npy + '/' + self.path_render_unlit_sscatter, render_unlit_sscatter)
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            for k in range(render_unlit_sscatter.shape[0]):
            # for k in range(70,80):
                render_sscatter = np.float32(render_unlit_sscatter[k])
                render_sscatter = np.power(render_sscatter, 1/self.gamma)
                if self.isMax:
                    utils.savePNG(path_png + '/render_sscatter/' + f'{str(k).zfill(4)}.png',render_sscatter/render_sscatter.max(), 'render_sscatter',self.isName)
                else:
                    utils.savePNG(path_png + '/render_sscatter/' + f'{str(k).zfill(4)}.png',render_sscatter, 'render_sscatter',self.isName)


        end_X09 = time.time()
        print('X09 time is: ', end_X09 - start_X09)


    def X02_renderSpecularComponent(self, path_root_object, path_view):
        start_X08 = time.time()

        path_npy = path_root_object + '/Show/npy_' + path_view
        fields = path_view.split('_')
        camera_index = int(fields[0].replace('Material', '')) - 37
        ##  预设参数
        map_normal = np.load(path_npy + '/' + self.path_map_normal)
        map_specular = np.load(path_npy + '/' + self.path_map_specular)
        render_unlit_sscatter = np.load(path_npy + '/' + self.path_render_unlit_sscatter)
        # lightDirs = np.load(self.path_lightDirs)
        lightDirs = np.load(self.path_lightDirsFront)
        viewDirs = np.load(self.path_viewDirs)
        result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        num_lights,height,width,channel = render_unlit_sscatter.shape
        print('result_first', result['Fit_First'])

        viewDir = viewDirs[camera_index, 0:]
        render_unlit_specular = utils.unlitSpecularRender(map_normal, map_specular, result, lightDirs, viewDir,
                                                          self.r0, self.lightIntensity,self.weight_shading)
        
        k_tmp = 1/render_unlit_specular.reshape(num_lights,-1).mean(-1) * render_unlit_sscatter.reshape(num_lights,-1).mean(-1)
        print('k_tmp',k_tmp)
        k_tmp_re = np.tile(k_tmp.reshape(num_lights,1,1,1),reps=(1,height,width,channel))
        render_unlit_specular = render_unlit_specular * k_tmp_re
        
        # wb_rs = result['Fit_First']['wb_rs']
        # render_unlit_specular = render_unlit_specular *wb_rs
        print('高光分量的最大值', np.max(render_unlit_specular))
        render_unlit_specular = utils.normalClipAndMax(render_unlit_specular, flag=self.flag)
        print('高光分量的最大值', np.max(render_unlit_specular))
        ## 保存无光高光渲染结果
        np.save(path_npy + '/' + self.path_render_unlit_specular, render_unlit_specular)

        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            for k in range(num_lights):
                render_specular = np.float32(render_unlit_specular[k, :, :, :])
                render_specular = np.power(render_specular, 1/self.gamma)
                if self.isMax:
                    utils.savePNG(path_png + '/render_specular/' + f'{str(k).zfill(4)}.png',render_specular/render_specular.max(), 'render_specular',self.isName)
                else:
                    utils.savePNG(path_png + '/render_specular/' + f'{str(k).zfill(4)}.png',render_specular, 'render_specular',self.isName)

        end_X08 = time.time()
        print('X08 time is: ', end_X08 - start_X08)


    def X02_renderDiffuseComponent(self, path_root_object, path_view):
        # path_root_object = os.path.join(self.root_path, path_object)
        start_X11 = time.time()
        path_npy = path_root_object + '/Show/npy_' + path_view

        map_diffuse = np.load(path_npy + '/' + self.path_map_diffuse)  # [h,w,c]
        render_unlit_shading = np.load(path_npy + '/' + self.path_render_unlit_shading)
        result = (np.load(path_npy + '/' + self.path_result, allow_pickle=True)).item()
        
        render_unlit_diffuse = utils.unlitDiffuseRender(map_diffuse,render_unlit_shading,self.ambientIntensity)
        
        wb_r = result['Fit_First']['wb_r']
        wb_g = result['Fit_First']['wb_g']
        wb_b = result['Fit_First']['wb_b']
        val = np.zeros_like(render_unlit_diffuse).astype('float32')
        val[:,:,:,0] = render_unlit_diffuse[:,:,:,0] *wb_r
        val[:,:,:,1] = render_unlit_diffuse[:,:,:,1] *wb_g
        val[:,:,:,2] = render_unlit_diffuse[:,:,:,2] *wb_b
        render_unlit_diffuse = val
        print('漫反射分量的最大值', np.max(render_unlit_diffuse))
        print('漫反射分量的平均值', np.mean(render_unlit_diffuse))
        render_unlit_diffuse = utils.normalClipAndMax(render_unlit_diffuse, flag=self.flag)
        print('漫反射分量的最大值', np.max(render_unlit_diffuse))
        print('漫反射分量的平均值', np.mean(render_unlit_diffuse))

        ## 保存漫反射渲染结果
        np.save(path_npy + '/' + self.path_render_unlit_diffuse, render_unlit_diffuse)
        
        if self.isSave:
            path_png = path_root_object + '/Show/image_' + path_view
            for k in range(render_unlit_shading.shape[0]):
            # for k in range(68,72):
                render_diffuse = np.float32(render_unlit_diffuse[k, :, :, :])
                render_diffuse = np.power(render_diffuse, 1/self.gamma)
                if self.isMax:
                    utils.savePNG(path_png + '/render_diffuse/' + f'{str(k).zfill(4)}.png',render_diffuse/render_diffuse.max(), 'render_diffuse',self.isName)
                else:
                    utils.savePNG(path_png + '/render_diffuse/' + f'{str(k).zfill(4)}.png',render_diffuse, 'render_diffuse',self.isName)

        end_X11 = time.time()
        print('X11 time is: ', end_X11 - start_X11)

