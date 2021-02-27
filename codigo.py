# Main Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset import
dataset = pd.read_csv('dados.csv')

# Idade do befeniciário (Masculinos)
x_masc = dataset.iloc[:26, 2:3].values
# Média do valor do beneficio por quantidade de beneficiários (Masculinos)
y_masc = dataset.iloc[:26, -2].values / dataset.iloc[:26, -3].values

# Sepração de dados de treinamento e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x_masc, y_masc, test_size = 0.15, random_state = 1)

# Cálculo da predição de dados
linearRegression = LinearRegression()
linearRegression.fit(x_treinamento, y_treinamento)

# Exibição com o matplotlib.pyplot dos dados de treinamento
plt.figure("Gráfico 01 - Treinamento")
plt.scatter(x_treinamento, y_treinamento, color="red")
plt.plot(x_treinamento, linearRegression.predict(x_treinamento), color="blue")
plt.title(f"Aposentadorias concedidas por anos de serviço (Para homens em 2018) \n Fórmula descrita por: f(x) = {linearRegression.coef_[0]:.2f}x + {linearRegression.intercept_:.2f}")
plt.xlabel("Anos de Experiência")
plt.ylabel("Valor médio do Benefício")
plt.show()

# Exibição com o matplotlib.pyplot dos dados de teste
plt.figure("Gráfico 02 - Teste")
plt.scatter(x_teste, y_teste, color="red")
plt.plot(x_teste, linearRegression.predict(x_teste), color="blue")
plt.title(f"Aposentadorias concedidas por anos de serviço (Para homens em 2018) \n Fórmula descrita por: f(x) =  {linearRegression.coef_[0]:.2f}x + {linearRegression.intercept_:.2f}")
plt.xlabel("Anos de Experiência")
plt.ylabel("Valor médio do Benefício")
plt.show()