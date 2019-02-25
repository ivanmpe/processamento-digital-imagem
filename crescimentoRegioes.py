import numpy as np
from collections import deque



 
class CrescimentoRegioes:
    
    def __init__(self, imagens):
        self.imagens = imagens
        
        
    def obterSemente(self, image):

        # A semente será um retângulo no centro
        
        w, h = image.shape
        p = 10. # p é uma porcentagem da altura e da largura
        
        pos_ini_x_mrk = int(w/2 - p*w/1000.)
        pos_ini_y_mrk = int(h/2 - p*h/1000.)
        pos_fim_x_mrk = int(w/2 + p*w/1000.)
        pos_fim_y_mrk = int(h/2 + p*h/1000.)
    
        # Semente é uma imagem do mesmo tamanho que img, contendo zeros
        semente = np.zeros(shape=(w,h), dtype=np.uint8)
        # acrescenta um retângulo central de pixels = 255
        semente[pos_ini_x_mrk:pos_fim_x_mrk, pos_ini_y_mrk:pos_fim_y_mrk] = 255
        
        return semente
    
    def vizinhos(self, x, y, w, h):
        
        lista = deque()
        
        pontos = [(x-1,y), (x+1, y), (x,y-1), (x,y+1),
                  (x-1,y+1), (x+1, y+1), (x-1,y-1), (x+1,y-1),
                 ]
        for p in pontos:
            if (p[0]>=0 and p[1]>=0 and p[0]<w and p[1]<h):
                lista.append((p[0], p[1]))
                
        return lista
        
    def crescerRegiao(self, image, reg, epsilon):
        
        
        w, h = image.shape
        
        fila = deque()
        for x in range(w):
            for y in range(h):
                if reg[x,y]==255:
                    fila.append((x,y))
           
        while fila:
            ponto = fila.popleft()
            x = ponto[0]
            y = ponto[1]
                
            v_list = self.vizinhos(x, y, w, h)
            for v in v_list:
                v_x = v[0]
                v_y = v[1]
                if( (reg[v_x][v_y]!=255) and (abs(image[x][y]-image[v_x][v_y])<epsilon)):
                    reg[v_x][v_y] = 255
                    fila.append((v_x,v_y))
            
        return reg
                    
    

        
            
        
