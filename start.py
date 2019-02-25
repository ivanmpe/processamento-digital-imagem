# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 00:02:31 2019

@author: Ivan Alves
"""

import cv2
import numpy as np
from skimage.io import imread
from scipy import signal as sg
from leituraImagens import LeituraImagens
from crescimentoRegioes import CrescimentoRegioes
from kmeans import Kmeans 
from time import time
import os
import matplotlib.pyplot as plt





def crescimentoRegioes(imagens, segmentadas,   path):
    #crescimento de Regioes 

    c = CrescimentoRegioes(imagens)
    qtd  = 230

    for i in imagens:

        img = imread( imagens[qtd][0], as_gray = True)
        img = (img * 255).round().astype(np.uint8)
        img = cv2.resize(img, (segmentadas[qtd][2], segmentadas[qtd][1]))   
        semente = c.obterSemente(img)
    
        media3 = [[1./9., 1./9., 1./9.], 
                  [1./9., 1./9., 1./9.], 
                  [1./9., 1./9., 1./9.]]
        
        c_media = sg.convolve( img , media3, "valid")
    
        kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (300, 300));
        blackhat = cv2.morphologyEx( c_media , cv2.MORPH_BLACKHAT, kernel)
        #cv2.imwrite("blackhat.jpg", blackhat)
        qtd +=1
        regiao = c.crescerRegiao( blackhat , semente, epsilon=3)
        cv2.imwrite(os.path.join(path, "segmentada_{}.png" .format(qtd)) , regiao)
        print("imagem: ", qtd)
    
    
def kmeans(imagens, segmentadas,   path):

    k = Kmeans(imagens)
    qtd  = 0
    
    for i in imagens: 
        # Leitura Imagem
        img = imread(imagens[qtd][0]) 
        #img = cv2.resize(img, (segmentadas[qtd][2], segmentadas[qtd][1]))   

        res2 = k.kmeans_seg(img, 2) / 255
        res3 = k.kmeans_seg(img, 3) / 255
        res9 = k.kmeans_seg(img, 5) / 255
        cv2.imwrite("res2.png", res2)
        cv2.imwrite("res3.png", res3)
        cv2.imwrite("res9.png", res9)

        fig = plt.figure(figsize=(9,3), dpi=200)
        k.add_image(fig, img, 1, 4, 1, 'original')
        k.add_image(fig, res2, 1, 4, 2, 'k=2')
        k.add_image(fig, res3, 1, 4, 3, 'k=3')
        k.add_image(fig, res9, 1, 4, 4, 'k=5')

if __name__ == '__main__':
 
    #leitura das Imagens
    start_time = time()
    imagens = []
    l = LeituraImagens
    imagens = l.leitura('../dataset/images/*/*/*.jpg')
    stop_time = time()
    tempo_total = stop_time - start_time
    print("Tempo total da leitura das imagens: ", tempo_total)

    
     #leitura das Imagens Segmentadas
    start_time = time()
    segmentadas = []
    s = LeituraImagens
    segmentadas = s.leitura('../dataset/segmented/*/*/*.png')
    stop_time = time()
    tempo_total = stop_time - start_time
    print("Tempo total da leitura das imagens segmentadas: ", tempo_total)


    #Crescimento de Regioes
    try:
        os.mkdir('./Crescimento Regioes')
    except OSError:
        print("Diretorio Crescimendo de Regioes já foi criado")
        
    path = './Crescimento Regioes'
    start_time = time()
    crescimentoRegioes(imagens, segmentadas, path)
    stop_time = time()
    tempo_total = stop_time - start_time
    print("Tempo total do crescimento de regiões: ", tempo_total)


    try:
        os.mkdir('./Kmeans')
    except OSError:
        print("Diretorio Kmeans ja foi criado")
    
    path = './Kmeans'
    start_time = time()
    kmeans(imagens, segmentadas, path)
    stop_time = time()
    tempo_total = stop_time - start_time
    print("Tempo total do kmeans: ", tempo_total)
    
    
"""    
    # Leitura Imagem
    #img1 = imread('3095_lg.tiff') 
    img1 = imread('t.jpg') 
    
    res2 = kmeans_seg(img1, 2) / 255
    res5 = kmeans_seg(img1, 5) / 255
    res9 = kmeans_seg(img1, 9) / 255
    
    fig = plt.figure(figsize=(9,3), dpi=200)
    add_image(fig, img1, 1, 4, 1, 'original')
    add_image(fig, res2, 1, 4, 2, 'k=2')
    add_image(fig, res5, 1, 4, 3, 'k=5')
    add_image(fig, res9, 1, 4, 4, 'k=9')
    
    
 """
    # Leitura Imagem
    img1 = imread('t.jpg') 
    
    res2 = kmeans_seg(img1, 2) / 255
    res3 = kmeans_seg(img1, 3) / 255
    res9 = kmeans_seg(img1, 5) / 255
    
    fig = plt.figure(figsize=(9,3), dpi=200)
    add_image(fig, img1, 1, 4, 1, 'original')
    add_image(fig, res2, 1, 4, 2, 'k=2')
    add_image(fig, res3, 1, 4, 3, 'k=3')
    add_image(fig, res9, 1, 4, 4, 'k=5')
        
        
    
    
    
    
    
    
    


