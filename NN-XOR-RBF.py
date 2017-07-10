# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:24:16 2017

Redes de Função de Base Radial

Solução do problema de classificação da função booleana "XOR" por meio
das Redes de Função de Base Radial, conforme Haykin (2001, págs. 311 a 313).

Referência: Haykin, S. Redes Neurais Artificiais - Princípios e Práticas.
2a Edição. Editora Bookman, 2001.
    
@author: Carlos Eduardo Véras Neves
"""
import numpy as np

# calcula a norma para a função Gaussiana
def norm(x):
    
    return np.sqrt(x[0]**2 + x[1]**2)

# calcula a função gaussiana para cada elemento da matriz
def gaussian(x, t):
    
    return np.exp(-(norm(x - t))**2)


if __name__ == '__main__':    
    
    # entradas da função booleana XOR
    x = np.array(((1.,1.),
                  (0.,1.),
                  (0.,0.),
                  (1.,0.)))
    
    print (" ### Entrada da Função XOR ###")
    print (x) 
    print("\n")
    
    
    # saídas - alvos - da  função booleana XOR
    d = np.array((0.,1.,0.,1.))
  
    # Matriz G de funções de Gaussianas com viés (b = 1.0)
    G = np.ones((x.shape[0],x.shape[1]+1))
    
    # centros das funções Gaussianas
    t = np.array(((1.,1.),
                  (0.,0.)))
    
    cols = x.shape[1]
    lins = x.shape[0]
    
    for i in range (lins):
        for j in range(cols):
            G[i][j] = gaussian(x[i], t[j])
    
    print (" ### Matriz de Funções Gaussianas ###")
    print (G) 
    print("\n")
    
    # calcula os pesos por meio da solução de norma mínima
    # w = G_plus*d = (G.T*G)^-1*G.T*d 
    G_plus = np.linalg.inv(np.dot(G.T, G))
    G_plus = np.dot(G_plus, G.T)
    w = np.dot(G_plus, d)
        
    print (" ### Vetor de Pesos ###")
    print (w)
    print("\n")
    
    # calcula as saídas da rede neural       
    y = np.dot(G, w)
    
    # imprime resultados
    print(" ### RESULTADOS ### ")
    
    i = 0
    for row in y:
        	
        print("Esperado=%f, Previsto=%f" % (d[i], row))
        i+=1
