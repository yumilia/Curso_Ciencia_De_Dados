# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:50:49 2021

@author: yumil
"""

# 1. Importe o numpy
import numpy as np

# 2. Crie x contendo um array em duas dimensões com os valores de Horas_de_Estudo já registrados no passado (vide tabela acima)
x = np.array([1,5,7,8,10,11,14,15,15,19]).reshape((-1,1))

# 3. Crie y contendo um array com os valores históricos da Nota já registrados no passado (vide tabela acima)
y = np.array([53,74,59,43,56,84,96,69,84,83])

# 4. Importe a biblioteca de regressão linear do sklearn
from sklearn.linear_model import LinearRegression

# 5. Crie o modelo de regressão linear, fazendo o fit de x e y.
modelo = LinearRegression().fit(x,y)

# 6. Avalie o modelo. Qual seu score, intercept e slope?
modelo.score(x,y)
modelo.intercept_
modelo.coef_

# 7. Crie novo_x contendo um array as seguintes horas de estudo dos novos alunos: 6, 9, 12, 15, 16, 4
novo_x = np.array([6,9,12,15,16,4]).reshape((-1,1))

# 8. Aplique o modelo criado fazendo uma previsão de notas no novo conjunto de dados (novo_x)
previsao = modelo.predict(novo_x)

# 9. Conseguiu descobrir quais serão as notas dos novos alunos? Imprima as notas dos novos alunos.
print(previsao)