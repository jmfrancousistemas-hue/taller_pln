Taller Procesamiento de Lenguaje Natural (PLN) y Modelos Fundacionales

Asignatura: Inteligencia Artificial
Estudiante: Juan Manuel Franco Rojas
Docente: Ing. Oscar F. Pulgarin

Modelo Implementado: Multi-Head Attention desde cero en PyTorch y análisis comparativo LSTM, GRU y Transformer.

Entorno de Ejecución: Python 3 con PyTorch, computadoras del laboratorio de sistemas.

GitHub: https://github.com/jmfrancousistemas-hue/taller_pln

Actividad 1: Análisis Crítico de Modelado Secuencial (Río Pescador)
1. Dimensionamiento del conjunto de datos
Frecuencia temporal: 1 lectura cada 10 minutos → 6 lecturas/hora × 24 horas/día = 144 lecturas/día.
Horizonte total (3 años): 144 × 365 × 3 = 157.680 pasos de tiempo (timesteps). Si se cuenta el día extra del año bisiesto (1.096 días en total), son 157.824 lecturas.
Naturaleza del problema: Serie temporal física univariada o multivariada de baja dimensionalidad (p. ej., caudal, precipitación pluvial, temperatura).
2. Selección de arquitectura recomendada: GRU (Gated Recurrent Unit) (con LSTM como alternativa)
Para este caso de estudio específico, una red recurrente con compuertas como GRU es más adecuada que un Transformer estándar o una RNN simple, por las siguientes razones:
Costo computacional y huella de memoria: Un Transformer convencional exhibe una complejidad cuadrática de atención O(N2) en la longitud de secuencia. Si se tomara la serie completa (157.680 pasos) como una sola secuencia, la matriz de atención tendría unos 2,5 × 10¹⁰ valores, cerca de 100 GB en float32 por cabeza y por capa, así que habría que cortarla en ventanas deslizantes o usar atención dispersa. En contraste, las compuertas recurrentes operan en complejidad O(Nd), lo que necesita mucha menos memoria y permite entrenar en los computadores normales del laboratorio.
Volumen y densidad de datos: 157680 puntos constituyen un volumen medio-pequeño para deep learning. Los Transformers carecen de sesgo inductivo temporal inherente (dependen enteramente de la codificación posicional aprendida o sinusoidal) y requieren grandes volúmenes de datos para no sobreajustar (overfitting). GRU y LSTM tienen un sesgo inductivo secuencial natural y suelen converger bien con este tamaño de muestra.
Latencia en inferencia y despliegue local: Para sistemas de alerta temprana o monitoreo agroindustrial continuo, inferir la predicción del siguiente intervalo de 10 minutos con una celda GRU toma fracciones de milisegundo en CPU. GRU unifica las compuertas de olvido y entrada y prescinde del estado de celda separado (ct), reduciendo los parámetros en un 25 % respecto a LSTM, con menor latencia y una precisión que en series hidrológicas suele ser comparable.
3. Comparación de las tres arquitecturas
Para comparar con números, se supone una entrada de 3 variables (caudal, precipitación y temperatura) y 64 neuronas ocultas. En PyTorch una LSTM tiene 4 bloques de compuertas de 64 × (3 + 64) pesos más dos sesgos de 64, es decir 4 × 4.416 = 17.664 parámetros; la GRU tiene 3 bloques, 13.248 parámetros, un 25 % menos.
Los dos valores se pueden comprobar en PyTorch con este fragmento, que usa las capas ya incluidas en la librería:

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

Resultado de la ejecución
uv run Actividad01.py

LSTM: 17664
GRU:  13248
Reduccion: 0.25

4. Conclusión
Recomiendo una GRU entrenada con ventanas deslizantes, por ejemplo las últimas 144 lecturas (un día) para predecir la lectura siguiente. Los datos se deben dividir en orden cronológico, sin mezclar fechas: dos años para entrenar, unos meses para validar y el último tramo para la prueba final. También conviene comparar el modelo contra una línea base simple, como repetir el último valor medido, para saber si la red de verdad aporta. Si en la validación la LSTM recuerda mejor los patrones largos, como las crecientes que se forman durante varios días, se puede cambiar sin tocar el resto del sistema. Un Transformer tendría sentido si más adelante se suman más estaciones, más variables o varios años adicionales de registros.
Actividad 2: Implementación de Multi-Head Attention desde cero en PyTorch
A continuación se presenta el módulo completo implementado sin recurrir a torch.nn.MultiheadAttention, siguiendo la formulación de Vaswani et al. (2017):

Attention(Q,K,V)=softmaxQKTdkV

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

Resultado de la ejecución
uv run AnalisisTorch.py

Entrada:           [8, 16, 64]
Salida:            [8, 16, 64]
Pesos de atencion: [8, 4, 16, 16]
Mascara causal:    OK
Parametros:        16640
Prueba exitosa: las dimensiones se conservan.

Explicación del código
La función atencion_escalada aplica la fórmula de Vaswani et al. (2017): multiplica Q por la transpuesta de K, divide entre la raíz de d_k para que los puntajes no crezcan con la dimensión y aplica softmax por filas. Así se obtienen pesos que suman 1 y que sirven para promediar los vectores de V.
La clase AtencionMultiCabeza proyecta la entrada con cuatro capas Linear (W_Q, W_K, W_V y W_O). Con d_model = 64 y 4 cabezas, cada cabeza trabaja con vectores de d_k = 16. El método separar_cabezas cambia la forma del tensor de [lote, n, 64] a [lote, 4, n, 16], de modo que las cuatro cabezas se calculan en una sola multiplicación de matrices; unir_cabezas hace el camino inverso, que corresponde a la concatenación de la fórmula.
La verificación usa un tensor aleatorio de forma [8, 16, 64]. La salida conserva esa forma, los pesos tienen forma [8, 4, 16, 16] (una matriz de 16 × 16 por cabeza) y cada fila de pesos suma 1. La segunda prueba aplica la máscara causal triangular del capítulo 4.2 y comprueba que el primer token no presta atención a los tokens que vienen después de él.
El módulo tiene 16.640 parámetros entrenables: cuatro matrices de 64 × 64 con sus sesgos, es decir 4 × (4.096 + 64). Para ejecutarlo en el laboratorio basta con instalar PyTorch (pip install torch) y correr python AnalisisTorch.py.

