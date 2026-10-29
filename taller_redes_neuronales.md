# INSTITUTO UNIVERSITARIA DE ROLDANILLO, VALLE – UNINTEP
## UNIDAD ACADÉMICA DE ELECTRICIDAD YSISTEMAS
### GUÍA PEDAGÓGICA Y MODULAR DE APRENDIZAJE: INTELIGENCIA ARTIFICIAL AVANZADA

---

| **Información Curricular** | **Detalle** |
| :--- | :--- |
| **Asignatura:** | Inteligencia Artificial |
| **Docente Titular:** | Oscar F. Pulgarin |
| **Nivel:** | Formación Universitaria / Ciclo Complementario |
| **Eje Temático:** | Procesamiento de Lenguaje Natural (PLN) y Modelos Fundacionales |
| **Ubicación Institucional:** | Sede Principal, Roldanillo (Valle del Cauca) |

---

## PRESENTACIÓN Y ENFOQUE DIDÁCTICO INSTITUCIONAL

En el marco de nuestra misión institucional orientada a la excelencia académica, la transferencia de tecnologías de frontera y la transformación productiva regional, presentamos este material de estudio formal. El propósito de este compendio es desmitificar y formalizar los fundamentos teóricos, matemáticos y computacionales detrás de los sistemas inteligentes modernos que procesan datos secuenciales e impulsan la revolución de la Inteligencia Artificial Generativa.

El recorrido pedagógico está estructurado de forma evolutiva bajo la estrategia didáctica **C-F-A** (*Concepto $\to$ Formalización $\to$ Aplicación Crítica*). Cada capítulo aborda las preguntas clave:
1. *¿Qué necesidad computacional motivó su origen?*
2. *¿Cuál es su formulación matemática y estructural?*
3. *¿Cuáles fueron sus límites y cómo abrieron paso a la siguiente iteración científica?*

---

# CAPÍTULO 1: REDES NEURONALES RECURRENTES (RNN, LSTM Y GRU)

### 1.1 Motivación y necesidad del modelado secuencial
Las redes neuronales densas (*feedforward*) asumen que todas las entradas $x^{(i)}$ y salidas $y^{(i)}$ son independientes e idénticamente distribuidas ($i.i.d.$). No obstante, en tareas del mundo real —como la predicción de series temporales agroclimáticas en el Valle del Cauca, el análisis de señales bioeléctricas o el procesamiento del lenguaje humano—, los datos poseen **dependencia temporal**: el significado de una palabra o el valor de una variable depende intrínsecamente de los elementos que le precedieron.

Para modelar secuencias de longitud variable $\mathbf{x} = (x_1, x_2, \dots, x_T)$, el sistema requiere un **estado interno u oculto** ($h_t$) que actúa como memoria de los pasos de tiempo anteriores.

---

### 1.2 Redes Recurrentes Vainilla (*Vanilla RNN*)
En cada instante temporal $t$, una RNN estándar toma como entrada el vector actual $x_t$ y el vector de estado oculto previo $h_{t-1}$, calculando el nuevo estado oculto $h_t$ mediante una transformación afín combinada seguida de una no linealidad:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)$$

Donde:
* $x_t \in \mathbb{R}^d$: Vector de características en el tiempo $t$.
* $h_t \in \mathbb{R}^m$: Estado latente o memoria contextual en el tiempo $t$.
* $W_{hh} \in \mathbb{R}^{m \times m}$: Matriz de pesos recurrente (compartida a lo largo del tiempo).
* $W_{xh} \in \mathbb{R}^{m \times d}$: Matriz de pesos de entrada.
* $W_{hy} \in \mathbb{R}^{c \times m}$: Matriz de pesos de salida hacia el espacio de clases $c$.
* $b_h, b_y$: Vectores de sesgo (*bias*).

#### Propiedad de Compartición de Parámetros
A diferencia de un perceptrón multicapa convencional, los pesos $(W_{hh}, W_{xh}, W_{hy})$ son **invariantes en el tiempo**. Esto permite que la red procese secuencias de longitud arbitraria sin disparar el número de parámetros entrenables.

```
       y_t-1                 y_t                 y_t+1
         ^                    ^                    ^
         |                    |                    |
     [W_hy]               [W_hy]               [W_hy]
         |                    |                    |
... -> [h_t-1] -- W_hh --> [ h_t ] -- W_hh --> [h_t+1] -> ...
         ^                    ^                    ^
         |                    |                    |
     [W_xh]               [W_xh]               [W_xh]
         |                    |                    |
       x_t-1                 x_t                 x_t+1
```

---

### 1.3 El algoritmo BPTT y la patología del gradiente
El entrenamiento de una RNN emplea **Backpropagation Through Time** (BPTT). Si consideramos la función de pérdida total $L = \sum_{t=1}^T L_t$, el gradiente respecto a la matriz $W_{hh}$ involucra la regla de la cadena a través de la historia temporal:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \left( \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} \right) \frac{\partial h_k}{\partial W_{hh}}$$

El término jacobiano $\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - h_j^2) W_{hh}^T$ se multiplica reiteradamente ($t - k$ veces):
1. **Desvanecimiento del gradiente (*Vanishing Gradient*):** Si los valores propios (*eigenvalues*) de $W_{hh}$ son menores a 1, o debido a la saturación de la función $\tanh$ (cuya derivada está en el intervalo $(0, 1]$), el gradiente decrece exponencialmente conforme la distancia temporal $t - k$ aumenta. En consecuencia, la red olvida dependencias a largo plazo.
2. **Explosión del gradiente (*Exploding Gradient*):** Si los valores propios son mayores a 1, la multiplicación iterativa dispara los gradientes a valores astronómicos ($NaN$ o infinito), desestabilizando el entrenamiento. Se mitiga empíricamente mediante *Gradient Clipping* (recorte de norma de gradiente):

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{si } \|\mathbf{g}\| \le \theta \\ \theta \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{si } \|\mathbf{g}\| > \theta \end{cases}$$

---

### 1.4 Redes con compuertas: LSTM (*Long Short-Term Memory*)
Diseñadas por Hochreiter y Schmidhuber (1997), las celdas LSTM resuelven el desvanecimiento del gradiente introduciendo una autopista de información aditiva: el **estado de la celda** ($c_t$), regulado por tres compuertas lógicas diferenciales fundamentadas en la función sigmoide ($\sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)$).

```
                 c_{t-1} --------------------[ x ]-----------------( + )------------> c_t
                                               ^                     ^          |
                                               | (f_t)        (i_t)  |          |
                                               |             [ x ]---+          |
                                               |              ^   ^             |
                                               |              |   | (c~_t)      |  (tanh)
                                               |              |  [tanh]         |    |
                                               |              |   |             |    v
                 h_{t-1} --+--[ \sigma ]-------+       [ \sigma ] |   [ \sigma ]-+- [ x ]---> h_t
                           |      (f)                      (i)             (o)         ^
                  x_t -----+-----------------------------------------------------------+ (o_t)
```

#### Ecuaciones constitutivas de la celda LSTM:
1. **Compuerta de olvido (*Forget Gate*):** Decide qué porcentaje de la memoria previa descartar:
   $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
2. **Compuerta de entrada (*Input Gate*):** Cuantifica qué nuevos valores se incorporarán a la memoria:
   $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
3. **Candidato de memoria (*Candidate Cell State*):** Genera nuevos vectores de información candidata:
   $$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
4. **Actualización del estado de la celda:** Combinación puramente lineal que esquiva la saturación no lineal:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
5. **Compuerta de salida (*Output Gate*) y cálculo de $h_t$:**
   $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(c_t)$$

*Nota didáctica:* $\odot$ denota el producto Hadamard (multiplicación elemento por elemento).

---

### 1.5 Celdas GRU (*Gated Recurrent Unit*)
Introducida por Cho et al. (2014), la unidad GRU unifica el estado oculto y la celda de memoria en una entidad única $h_t$, reduciendo la carga paramétrica y el tiempo de convergencia:
* **Compuerta de actualización (*Update Gate*):** $z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$
* **Compuerta de reinicio (*Reset Gate*):** $r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$
* **Estado candidato:** $\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b)$
* **Estado final:** $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

---

# CAPÍTULO 2: MODELOS DE LENGUAJE BASADOS EN ATENCIÓN SOBRE RNN

### 2.1 El paradigma Codificador-Decodificador (*Encoder-Decoder / Seq2Seq*)
En tareas como la traducción automática o el resumen de textos, la longitud de la entrada $T_x$ no necesariamente coincide con la de la salida $T_y$. Sutskever et al. y Cho et al. propusieron la arquitectura Seq2Seq:
* **Codificador (RNN):** Procesa la secuencia de entrada $\mathbf{x} = (x_1, \dots, x_{T_x})$ y la comprime en un único vector de contexto fijo $\mathbf{c} = h_{T_x}$.
* **Decodificador (RNN):** Inicializa su estado oculto con $\mathbf{c}$ y genera secuencialmente la salida prediciendo $y_t$ condicionado a $(y_{1}, \dots, y_{t-1}, \mathbf{c})$.

```
           [ x_1 ] -> [ x_2 ] -> ... -> [ x_Tx ]
              |          |                 |
Codificador: (RNN) ---> (RNN) ---------> (RNN) ====> [ Vector c ] (Cuello de botella)
                                                           ||
Decodificador:                                    [ s_0 = c ] ---> (RNN) -> y_1
                                                                    |
                                                                  (RNN) -> y_2
```

#### El problema del cuello de botella (*Information Bottleneck*)
Comprimir un párrafo de 50 palabras con riqueza semántica y sintáctica en un único vector estático $\mathbf{c} \in \mathbb{R}^m$ provoca una severa degradación en el desempeño conforme aumenta $T_x$. La red sufre de "amnesia contextual" respecto al inicio de la frase.

---

### 2.2 El mecanismo de atención de Bahdanau (Atención Aditiva)
En 2014, Dzmitry Bahdanau, Kyunghyun Cho y Yoshua Bengio revolucionaron el modelado secuencial al postular que **el vector de contexto no debe ser estático, sino dinámico para cada instante de decodificación $t$**:

$$\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{ti} h_i$$

Donde $h_i$ son todas las anotaciones o estados ocultos generados por el codificador bidireccional, y $\alpha_{ti}$ representa la distribución de pesos de atención sobre la entrada.

#### Algoritmo de cálculo de atención:
1. **Puntaje de alineación (*Score / Energy*):** Evalúa la afinidad entre el estado previo del decodificador $s_{t-1}$ y el estado del codificador $h_i$:
   $$e_{ti} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$$
   *(Parámetros entrenables: $v_a, W_a, U_a$)*
2. **Normalización por Softmax:** Convierte las energías en una distribución de probabilidad válida:
   $$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T_x} \exp(e_{tk})}, \quad \sum_{i=1}^{T_x} \alpha_{ti} = 1$$
3. **Cálculo del vector de contexto dinámico $\mathbf{c}_t$:** Promedio ponderado de los estados ocultos.
4. **Cálculo del nuevo estado del decodificador:**
   $$s_t = f(s_{t-1}, y_{t-1}, \mathbf{c}_t)$$

```
  Codificador (h_1, h_2, ..., h_Tx)
         \     |     /
          v    v    v
     +-------------------+
     | Mecanismo de      | <--- Estado decodificador previo (s_t-1)
     | Atención Dinámica |
     +-------------------+
               |
               v
     Vector Contexto (c_t) ----> [ Decodificador s_t ] ---> Predicción (y_t)
```

---

### 2.3 Atención Multiplicativa (Luong et al., 2015)
Minh-Thang Luong propuso variantes donde la función de puntuación utiliza operaciones matriciales más eficientes:
* **Producto punto directo (*Dot*):** $e_{ti} = s_t^T h_i$ (asume igual dimensión en codificador y decodificador).
* **General:** $e_{ti} = s_t^T W_a h_i$.
* **Concatenación:** $e_{ti} = v_a^T \tanh(W_a [s_t; h_i])$.

### 2.4 El límite infranqueable de las RNN con Atención
A pesar del éxito del mecanismo de atención acoplado a RNNs, persistía un problema estructural inherente: **la naturaleza secuencial del cómputo**. Para calcular $h_t$, es mandatorio haber calculado $h_{t-1}$. Esto impide la paralelización masiva en hardware acelerador (GPU / TPU), convirtiendo el entrenamiento en conjuntos masivos de datos textuales en una tarea computacionalmente prohibitiva ($O(N)$ operaciones secuenciales).

---

# CAPÍTULO 3: MODELOS DE LENGUAJE BASADOS EN TRANSFORMERS

### 3.1 La ruptura epistemológica: "Attention Is All You Need"
En 2017, Vaswani et al. (Google Brain / Google Research) formularon un postulado radical: eliminar por completo la recurrencia y las convoluciones, estructurando la arquitectura únicamente mediante mecanismos de atención sobre vectores posicionales. Nace el **Transformer**.

---

### 3.2 Formalización del mecanismo de Auto-Atención Escalada (*Scaled Dot-Product Attention*)
El mecanismo mapea un conjunto de consultas (*Queries*, $\mathbf{Q}$), claves (*Keys*, $\mathbf{K}$) y valores (*Values*, $\mathbf{V}$) hacia una salida ponderada.

Para una secuencia de entrada con matriz de incrustación $X \in \mathbb{R}^{n \times d_{\text{model}}}$, se proyecta linealmente:
$$\mathbf{Q} = X W_Q, \quad \mathbf{K} = X W_K, \quad \mathbf{V} = X W_V$$
Donde $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ y $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

La ecuación de auto-atención escalada es:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

#### Justificación didáctica del factor de escalamiento $\frac{1}{\sqrt{d_k}}$
Si componentes de $q$ y $k$ son variables aleatorias independientes con media 0 y varianza 1, el producto punto $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ tiene media 0 y varianza $d_k$. Para valores grandes de $d_k$, las magnitudes del producto escalar crecen enormemente, empujando la función $\text{softmax}$ hacia regiones con derivadas extremadamente diminutas (saturación), provocando desvanecimiento del gradiente. Dividir por $\sqrt{d_k}$ preserva la varianza unitaria.

---

### 3.3 Atención de Cabezas Múltiples (*Multi-Head Attention - MHA*)
En lugar de calcular una sola función de atención con dimensión $d_{\text{model}}$, se proyectan las consultas, claves y valores $h$ veces con transformaciones lineales independientes a dimensiones $d_k, d_k, d_v$.

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{donde } \text{head}_i = \text{Attention}(\mathbf{Q} W_i^Q, \mathbf{K} W_i^K, \mathbf{V} W_i^V)$$

*Impacto conceptual:* Cada cabeza se especializa en capturar distintos tipos de relaciones sintácticas y semánticas simultáneamente (por ejemplo: una cabeza monitorea dependencias sujeto-verbo, otra correferencias pronominales y otra concordancia de género/número).

```
                  Q, K, V
                     |
       +-------------+-------------+
       |             |             |
   Proyección    Proyección    Proyección
   Lineal (1)    Lineal (2)    Lineal (h)
       |             |             |
   Attention     Attention     Attention
    (head 1)      (head 2)      (head h)
       \             |             /
        +------------+------------+
                     |
               Concatenación
                     |
             Proyección Final W^O
```

---

### 3.4 Codificación Posicional (*Positional Encoding*)
Al no poseer recurrencia, la arquitectura Transformer es invariante a la permutación: si permutamos el orden de las palabras en una frase, los cálculos de atención producirían exactamente los mismos valores numéricos desordenados. Para inyectar el orden sintáctico, se suma un vector posicional $P \in \mathbb{R}^{n \times d_{\text{model}}}$ a las incrustaciones de entrada:

$$X_{\text{input}} = \text{Embedding}(X) + P$$

Vaswani et al. propusieron funciones sinusoidales de diferentes frecuencias:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

Donde $pos$ es la posición de la palabra en la secuencia e $i$ es el índice de la dimensión. Esto permite al modelo extrapolar distancias relativas, pues para cualquier desfase $\Delta k$, existe una transformación lineal que proyecta $PE_{pos + \Delta k}$ a partir de $PE_{pos}$.

---

### 3.5 Arquitectura completa: Capas feed-forward, residuales y normalización
Cada bloque de la arquitectura Transformer implementa:
1. **Conexiones Residuales (*Add*):** Inspiradas en ResNet, mitigan la degradación del gradiente en arquitecturas profundas: $x + \text{Sublayer}(x)$.
2. **Normalización de Capa (*LayerNorm* / RMSNorm):** Normaliza sobre el espacio de características del vector en lugar de calcular estadísticas sobre el mini-lote (*BatchNorm*).
3. **Redes Feed-Forward Puntuales (*Position-Wise Feed-Forward Networks*):** Dos transformaciones lineales con una activación no lineal (ReLU, GeLU o SwiGLU) aplicadas idénticamente a cada posición:
   $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

---

### 3.6 Taxonomía de los Transformers según su configuración
Dependiendo de su diseño estructural, la familia de modelos Transformers se bifurca en tres vertientes:

```
                          ARQUITECTURAS TRANSFORMER
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
  Solo Codificador             Codificador-Decodificador     Solo Decodificador
  (Auto-encodificadores)       (Seq2Seq Original)           (Autorregresivos)
  Ej: BERT, RoBERTa            Ej: T5, BART                 Ej: GPT, LLaMA, Claude
  Uso: Clasificación,          Uso: Traducción,             Uso: Generación de texto,
  Extracción NER               Resumen abstracto            Razonamiento, Chatbots
```

---

# CAPÍTULO 4: INTELIGENCIA ARTIFICIAL GENERATIVA

### 4.1 De los modelos discriminativos a los generativos
* **Enfoque Discriminativo:** Modela la distribución condicional de probabilidad $P(Y \mid X)$. Su objetivo es clasificar, etiquetar o discriminar entre categorías conocidas (ej. determinar si un correo institucional es spam o no).
* **Enfoque Generativo:** Modela la distribución de probabilidad conjunta $P(X, Y)$ o directamente la densidad no condicional $P(X)$. Su objetivo es aprender la estructura estadística latente de los datos para muestrear y sintetizar **nuevas instancias plausibles** $x_{\text{nuevo}} \sim P_{\text{modelo}}(X)$ que no existían previamente en la base de datos de entrenamiento.

---

### 4.2 Modelado de Lenguaje Autorregresivo (Causal Language Modeling - CLM)
Los Grandes Modelos de Lenguaje (LLMs) actuales se entrenan bajo el objetivo de predicción del siguiente token (*Next-Token Prediction*), aplicando la regla de la cadena de probabilidades:

$$P(\mathbf{x}) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Durante el entrenamiento, se minimiza la pérdida de entropía cruzada (*Cross-Entropy Loss*):

$$\mathcal{L}_{\text{CLM}}(\theta) = - \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

Para impedir que el decodificador atienda a tokens futuros durante el entrenamiento, se implementa una **Máscara Causal Triangulada Superior** en la matriz de auto-atención antes del softmax, rellenando con $-\infty$ las posiciones $j > i$.

```
Máscara Causal (Look-Ahead Mask):
[  0   -\infty  -\infty ]
[  0      0     -\infty ]
[  0      0        0    ]
```

---

### 4.3 Estrategias de muestreo y decodificación (*Decoding Strategies*)
Una vez que el modelo produce una distribución de probabilidad sobre el vocabulario $V$, la selección de los tokens generados define la fluidez, coherencia y creatividad:
1. **Búsqueda Codiciosa (*Greedy Search*):** Selecciona siempre $\text{argmax}_{w} P(w \mid x_{<t})$. Tiende a generar repeticiones en bucle.
2. **Temperatura ($T$):** Modifica la escala de los *logits* $z_i$:
   $$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   * $T \to 0$: Comportamiento determinista, rígido y conservador.
   * $T > 1$: Distribución uniforme, fomenta creatividad pero incrementa riesgo de alucinación.
3. **Top-$k$ Sampling:** Restringe el muestreo a los $k$ tokens más probables.
4. **Top-$p$ Sampling (Nucleus Sampling):** Selecciona el subconjunto más pequeño de tokens cuya masa acumulada supera el umbral probabilístico $p$ ($\sum_{w \in V^{(p)}} P(w) \ge p$).

---

### 4.4 Ciclo de vida y alineación de LLMs modernos
Para transformar un modelo base (que solo predice la continuación estadística de un texto) en un asistente útil, veraz y seguro, se aplica un protocolo estricto de tres etapas:

```
[ Preentrenamiento Masivo ]
(Auto-supervisado con billones de tokens -> Modelo Base / Foundation Model)
                 |
                 v
[ Ajuste Fino Instruccional (SFT) ]
(Supervised Fine-Tuning con pares Prompt-Respuesta de alta calidad)
                 |
                 v
[ Alineación con Preferencias Humanas ]
(RLHF / DPO: Direct Preference Optimization / PPO)
                 |
                 v
[ LLM Listo para Despliegue Seguro ]
```

1. **Preentrenamiento (*Pre-training*):** Aprendizaje autosupervisado masivo sobre terabytes de datos web, libros y código. Extrae conocimiento del mundo y patrones sintácticos.
2. **Ajuste Fino Instruccional (*Instruction Tuning / SFT*):** Entrenamiento supervisado con miles de ejemplos curados por expertos donde el modelo aprende a responder como asistente.
3. **Alineación con Preferencias Humanas (RLHF / DPO):**
   * **RLHF (*Reinforcement Learning from Human Feedback*):** Entrena un modelo de recompensa (*Reward Model*) y optimiza la política del LLM mediante algoritmos de gradiente como PPO (*Proximal Policy Optimization*).
   * **DPO (*Direct Preference Optimization*):** Optimiza directamente la política sobre pares de respuestas preferidas y rechazadas sin requerir un modelo de recompensa explícito ni entrenamiento por refuerzo inestable.

---

### 4.5 Otros paradigmas de IA Generativa en el ecosistema actual
Aunque los Transformers dominan el texto y el código, la IA Generativa abarca múltiples arquitecturas para audio, imagen y video:
* **Redes Generativas Antagónicas (GANs):** Juego de teoría de juegos bi-personal de suma cero entre un Generador $G$ y un Discriminador $D$.
* **Autoencoders Variacionales (VAEs):** Modelado probabilístico con regularización por divergencia Kullback-Leibler sobre un espacio latente estructurado.
* **Modelos de Difusión (*Denoising Diffusion Probabilistic Models - DDPM*):** Añaden ruido gaussiano de forma iterativa (*proceso hacia adelante*) y aprenden mediante redes tipo U-Net o Diffusion Transformers (DiT) a invertir el proceso eliminando dicho ruido paso a paso (*proceso inverso*), liderando la generación fotorrealista (Midjourney, Stable Diffusion, Sora).

---

# SÍNTESIS COMPARATIVA FINAL

| Dimensión | Redes Recurrentes (RNN/LSTM) | RNN con Mecanismo de Atención | Transformers (Vaswani et al.) | LLMs Generativos Modernos |
| :--- | :--- | :--- | :--- | :--- |
| **Complejidad Secuencial** | $O(N)$ (Estrictamente paso a paso) | $O(N)$ (Secuencial en ejecución) | $O(1)$ (Paralelización total temporal) | $O(1)$ entrenamiento, autorregresivo en inferencia |
| **Complejidad de Memoria** | $O(N \cdot d)$ | $O(N \cdot d)$ | $O(N^2)$ (Atención estándar cuadrática) | Optimizada mediante FlashAttention / KV-Cache |
| **Modelado de Contexto** | Memoria finita; sufre atenuación | Conexión directa a estados previos | Global, directo para cualquier par de tokens | Ventanas de contexto extensas ($32k - 1M+$ tokens) |
| **Cuello de Botella** | Vector oculto estático $h_t$ | Parcialmente aliviado | Ausente a nivel estructural | Escalado de cómputo y consumo energético |
| **Objetivo Central** | Predicción paso a paso | Traducción / Mapeo Seq2Seq | Representaciones contextuales densas | Generación multimodal y razonamiento emergente |

---

## TALLER DE APLICACIÓN DIDÁCTICA Y AUTOEVALUACIÓN

Para afianzar los conceptos abordados en las sesiones teóricas de nuestra facultad, desarrolle las siguientes actividades orientadas:

1. **Análisis Crítico:** Suponga que la Corporación Autónoma Regional o una empresa agroindustrial del norte del Valle del Cauca le solicita modelar el flujo de caudal hídrico del río Pescador con lecturas cada 10 minutos durante 3 años. ¿Implementaría una LSTM, una GRU o un Transformer? Argumente su respuesta considerando costo computacional, volumen de datos y requerimientos de latencia.
2. En las computadoras del laboratorio de sistemas de UNINTEP, implemente en PyTorch un módulo básico de Multi-Head Attention desde cero (sin invocar `torch.nn.MultiheadAttention`) y verifique la conservación de dimensiones mediante tensores sintéticos.

---
**Documento de Estudio Institucional - UNINTEP**  
*Comprometidos con la excelencia académica, la ciencia y la innovación tecnológica.*