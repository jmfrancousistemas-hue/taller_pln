import torch.nn as nn


# Misma configuracion de la tabla: 3 variables de entrada, 64 neuronas ocultas
lstm = nn.LSTM(input_size=3, hidden_size=64)
gru = nn.GRU(input_size=3, hidden_size=64)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


n_lstm = contar_parametros(lstm)
n_gru = contar_parametros(gru)
print("LSTM:", n_lstm)                    # 17664
print("GRU: ", n_gru)                     # 13248
print("Reduccion:", 1 - n_gru / n_lstm)   # 0.25 -> 25 % menos