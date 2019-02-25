# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:05:19 2019

@author: Ivan Alves
"""

import glob
import cv2

class LeituraImagens:
    
    def leitura( diretorio ):
        imagens = []
        caminhos = glob.glob(diretorio)
        qtdeImagens=0
        
        for caminho in caminhos:
            qtdeImagens +=1
            tamanho = cv2.imread(caminho).shape
            imagens.append((caminho, tamanho[0], tamanho[1]))
        
        print("Total de: {} imagens " .format(qtdeImagens))
        return imagens
    

