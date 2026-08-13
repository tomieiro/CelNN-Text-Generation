# Etapa 1 — Gate de capacidade no Text8

## Objetivo

Medir se o aumento de capacidade do núcleo CelNN reduz a diferença de qualidade
para um controle convolucional convencional sob o mesmo protocolo de treino. O
gate foi definido antes da execução: uma configuração CelNN passa se terminar a
no máximo 0,25 BPC da rung H após 20.000 passos e três seeds.

## Protocolo

- Dataset: Text8, modelagem em nível de caractere.
- Treino: 20.000 passos por seed.
- Seeds: 42, 1337 e 2024.
- Hardware: NVIDIA A100-SXM4-80GB.
- Software: PyTorch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2 e libPyCelNN no
  commit `1685319256e5f167ea1fef9d5f9e48a7e8b69578`.
- Execução: job SLURM 11831, com três seeds concorrentes por rung.
- Métrica principal: bits por caractere (BPC), em que menor é melhor.

## Resultados

| Rung | Tipo | BPC médio | Desvio | Parâmetros do núcleo | Parâmetros totais | Operações analíticas | Latência GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | CelNN | 2,7863 | 0,0080 | 7 | 3.490 | 1.032.192 | 33,45 ms |
| B | CelNN | 2,7397 | 0,0088 | 896 | 4.379 | 1.032.192 | 33,67 ms |
| C | CelNN | 2,6559 | 0,0042 | 1.920 | 5.403 | 3.129.344 | 21,27 ms |
| D | CelNN | 2,5655 | 0,0142 | 2.944 | 6.427 | 5.226.496 | 21,63 ms |
| E | CelNN | 2,4778 | 0,0044 | 4.992 | 8.475 | 9.420.800 | 20,76 ms |
| F | CelNN | 2,3733 | 0,0066 | 9.088 | 12.571 | 17.809.408 | 21,32 ms |
| G | CelNN | **2,2802** | 0,0093 | 17.280 | 20.763 | 34.586.624 | 21,04 ms |
| H | Controle convolucional | **1,8312** | 0,0010 | 394.240 | 397.723 | 25.387.008 | 0,64 ms |

## Achados

1. A qualidade das variantes CelNN melhora monotonicamente de A até G. O ganho
   acumulado é de 0,5062 BPC, de 2,7863 para 2,2802.
2. A melhor variante CelNN é a rung G, mas ela termina 0,4490 BPC atrás do
   controle H. Essa diferença excede em 0,1990 BPC o limite pré-definido.
3. Portanto, **a Etapa 1 não passa o gate de capacidade**. No desenho e orçamento
   atuais, aumentar somente a capacidade do núcleo CelNN não fecha a diferença
   para o controle convolucional.
4. Não há knee elegível pela política original, pois nenhuma rung CelNN passa o
   gate. A melhora A→B é menor que 0,05 BPC, mas isso isoladamente não torna A
   ou B candidatas ao knee.
5. A baixa dispersão entre seeds indica que a ordenação observada é estável
   neste protocolo. A maior dispersão foi 0,0142 BPC na rung D.
6. O valor de 0,64 ms medido para H é muito menor que os aproximadamente 21–34
   ms das demais rungs. Essa diferença deve ser tratada como **não validada** até
   repetir o benchmark com sincronização CUDA explícita, warm-up e distribuição
   de múltiplas medições. Ela não sustenta, nesta etapa, uma conclusão de
   desempenho em tempo de inferência.

## Integridade e artefatos

O job atingiu o limite de oito horas depois de gravar o JSON consolidado, o
notebook executado e os 24 checkpoints finais (oito rungs por três seeds). A
consulta tardia retornou `UNKNOWN` porque o job já não estava disponível no
accounting do SLURM; o log remoto registra o timeout, mas os artefatos completos
foram recuperados diretamente do diretório remoto.

Artefatos locais:

- `experiment/.dgx-results/11831/experiment-0.json` — resultados consolidados;
- `experiment/.dgx-results/11831/checkpoints/` — 24 checkpoints finais;
- `experiment/experiment-0-gpu.executed.ipynb` — notebook executado.

## Decisão para a próxima etapa

A próxima etapa não deve apenas ampliar a mesma escada de capacidade. Ela deve
investigar a diferença estrutural entre a dinâmica CelNN e o controle H, além de
validar separadamente o benchmark de latência GPU. Os checkpoints desta etapa
permitem análise e inferência sem repetir o treinamento concluído.
