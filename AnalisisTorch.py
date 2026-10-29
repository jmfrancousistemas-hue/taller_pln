# Taller PLN - Actividad 2: Multi-Head Attention desde cero en PyTorch
# Estudiante: Juan Manuel Franco Rojas - UNINTEP

import math
import torch
import torch.nn as nn


def atencion_escalada(Q, K, V, mascara=None):
    """Attention(Q, K, V) = softmax(Q K^T / raiz(d_k)) V"""
    d_k = Q.size(-1)
    # Puntajes: que tanto se parece cada consulta a cada clave
    puntajes = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Mascara opcional: donde vale 0 se pone un numero muy negativo
    if mascara is not None:
        puntajes = puntajes.masked_fill(mascara == 0, -1e9)
    # Softmax por filas: cada fila de pesos suma 1
    pesos = torch.softmax(puntajes, dim=-1)
    return torch.matmul(pesos, V), pesos


class AtencionMultiCabeza(nn.Module):
    def __init__(self, d_model, num_cabezas):
        super().__init__()
        assert d_model % num_cabezas == 0, "d_model debe ser divisible por num_cabezas"
        self.num_cabezas = num_cabezas
        self.d_k = d_model // num_cabezas  # dimension de cada cabeza

        # Matrices W_Q, W_K, W_V y la proyeccion final W_O
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def separar_cabezas(self, x):
        # [lote, n, d_model] -> [lote, cabezas, n, d_k]
        lote, n, _ = x.shape
        return x.view(lote, n, self.num_cabezas, self.d_k).transpose(1, 2)

    def unir_cabezas(self, x):
        # [lote, cabezas, n, d_k] -> [lote, n, d_model]
        lote, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().view(lote, n, self.num_cabezas * self.d_k)

    def forward(self, q, k, v, mascara=None):
        # 1. Proyectar y repartir en cabezas
        Q = self.separar_cabezas(self.W_q(q))
        K = self.separar_cabezas(self.W_k(k))
        V = self.separar_cabezas(self.W_v(v))
        # 2. Atencion escalada en todas las cabezas a la vez
        contexto, pesos = atencion_escalada(Q, K, V, mascara)
        # 3. Concatenar las cabezas y aplicar W_O
        salida = self.W_o(self.unir_cabezas(contexto))
        return salida, pesos


if __name__ == "__main__":
    torch.manual_seed(0)  # para que los resultados se repitan
    lote, n, d_model, cabezas = 8, 16, 64, 4

    # Tensor sintetico: 8 secuencias de 16 tokens con vectores de 64 numeros
    x = torch.randn(lote, n, d_model)
    mha = AtencionMultiCabeza(d_model, cabezas)

    # Prueba 1: auto-atencion (q = k = v = x)
    salida, pesos = mha(x, x, x)
    print("Entrada:          ", list(x.shape))
    print("Salida:           ", list(salida.shape))
    print("Pesos de atencion:", list(pesos.shape))
    assert salida.shape == x.shape
    assert pesos.shape == (lote, cabezas, n, n)
    # Cada fila de pesos debe sumar 1 (propiedad del softmax)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(lote, cabezas, n), atol=1e-5)

    # Prueba 2: mascara causal (un token no ve los tokens futuros)
    causal = torch.tril(torch.ones(n, n))
    _, pesos_c = mha(x, x, x, mascara=causal)
    assert pesos_c[0, 0, 0, 1:].sum() < 1e-6  # el primer token solo se ve a si mismo
    print("Mascara causal:    OK")

    # Numero de parametros entrenables: 4 * (64*64 + 64) = 16.640
    total = sum(p.numel() for p in mha.parameters())
    print("Parametros:       ", total)
    print("Prueba exitosa: las dimensiones se conservan.")