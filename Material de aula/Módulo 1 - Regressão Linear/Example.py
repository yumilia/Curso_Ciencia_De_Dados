# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:39:25 2021

@author: yumil
"""
import numpy as np
from sklearn.linear_model import LinearRegression

# 2. Cria o dataset. Criamos em formato de array. O reshape cria em duas dimensões (uma coluna e várias linhas)
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

modelo = LinearRegression()
modelo.fit(x,y)

# ... ou assim em uma linha
modelo = LinearRegression().fit(x,y)

# 4. Avalia o modelo
print('coeficiente de determinação:', modelo.score(x, y))

# Intercept
print('intercept:', modelo.intercept_)

# Slope
print('slope:', modelo.coef_)

# 5. Cria um novo conjunto de dados x. Arange gera um array com elementos de 0(inclusivo) a 5 (exclusivo)
novo_x = np.arange(5).reshape((-1, 1))
print(novo_x)

# 6. Aplica o modelo num novo conjunto de dados
previsao_y = modelo.predict(novo_x)
print(previsao_y)

# Outra forma idêntica de aplicar o modelo para prever resultados
previsao_y = modelo.intercept_ + modelo.coef_ * novo_x
print('previsão:', previsao_y)
