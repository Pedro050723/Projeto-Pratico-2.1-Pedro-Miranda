# Redes Neurais Artificiais – Projeto Prático 2.1
## Aprendizado supervisionado no neurônio Perceptron

**Aluno:** Pedro Henrique Miranda &nbsp;|&nbsp; **Trabalho individual** &nbsp;|&nbsp; **ID da equipe:** 8 % 4 = 0
**Disciplina:** Redes Neurais Artificiais &nbsp;|&nbsp; **Professor(a):** Elloá B. Guedes &nbsp;|&nbsp; **Instituição:** UEA

Implementação do neurônio Perceptron de Rosenblatt em Python, **do zero** (somente `numpy` nos cálculos do algoritmo), com experimentação de hiperparâmetros e validação *holdout*. Todo o projeto está em um único notebook, já executado, com as saídas visíveis.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `Projeto_Pratico2_1_RNA_PEDRO_HENRIQUE_MIRANDA.ipynb` | Notebook com as três partes, código e análises |
| `dataAll20262.txt` | Dados da Parte I (problema linearmente separável) |
| `data020262.txt` | Dados da Parte II (arquivo do ID 0) |
| `dataHoldout20262.txt` | Dados da Parte III (problema não linearmente separável) |

Os arquivos `.txt` são, na verdade, arrays do numpy salvos com `np.save`, e por isso são lidos com `np.load`.

## O que cada parte faz

**Parte I – Perceptron em dados linearmente separáveis.** Treino com η = 0,1, ativação degrau (limiar 0) e pesos iniciais sorteados em U(−0,5, +0,5). Reporta o número total de ajustes e de épocas até a convergência e plota a reta de separação.

**Parte II – Experimentação.** Avalia 6 configurações, η ∈ {0,4; 0,1; 0,01} × intervalo dos pesos iniciais ∈ {(−100, +100); (−0,5, +0,5)}, com 10 repetições cada. Para cada configuração, calcula média e desvio padrão do nº de ajustes e mínimo e máximo de épocas, apresenta uma tabela e um boxplot, e discute se há configuração melhor ou pior.

**Parte III – Validação *holdout*.** Divisão 70% treino / 30% teste com `np.random.permutation` e semente igual ao ID, treino por 100 épocas com exemplos embaralhados a cada época, e avaliação por matriz de confusão, acurácia, precisão, revocação e F-score, com gráficos de treino e teste lado a lado e discussão da qualidade da solução.

## Decisões de método

- **Função de treino única.** A mesma função, `treinar_perceptron`, é usada nas três partes. As opções (`max_epocas`, `embaralhar`) mudam o comportamento conforme a parte.
- **Normalização na Parte II.** Com os dados originais (valores próximos de ±1000), o treino não convergiu nas execuções testadas. As entradas foram padronizadas (*z-score*, com o *bias* adicionado depois) e o treino roda sem limite de épocas até convergir. Isso altera o significado de η e dos intervalos de pesos, então as conclusões da Parte II valem para os dados normalizados.
- **Acurácia na Parte III.** O enunciado é ambíguo sobre o conjunto da questão 2 (acurácia "para os dados do conjunto de treinamento"), por isso são reportadas as acurácias de treino e de teste, além de uma linha de base (sempre prever a classe majoritária do treino).
- **Bibliotecas.** `numpy`, `matplotlib`, `pandas` (somente para apresentar tabelas), `prettytable` (tabelas de resultados) e `sklearn.metrics` (somente para conferir as métricas da Parte III). Matriz de confusão, acurácia, precisão, revocação e F-score foram calculadas manualmente e conferidas com o `sklearn`. `train_test_split` não foi usado.

## Resultados principais

- **Parte I:** convergência em poucas épocas (na execução do notebook, 41 ajustes e 3 épocas; sem semente fixa, os valores variam a cada execução).
- **Parte II:** todas as 60 execuções convergiram. A única configuração claramente pior foi η = 0,01 com pesos iniciais em (−100, +100), com média de 7728 ajustes, cerca de 14 vezes a menor média. As outras cinco, com médias entre 543 e 1305 ajustes, são estatisticamente equivalentes com 10 repetições.
- **Parte III:** acurácia de 63,29% no treino e 63,00% no teste, contra 46,00% da linha de base. A precisão é alta (0,98 no teste) e a revocação é baixa (0,32), ou seja, a reta erra principalmente por omitir exemplos da classe 1. As métricas parecidas entre treino e teste indicam que o limite está no modelo (uma única reta) e no treinamento, e não na generalização.

## Como executar

**No Google Colab.** Abra o notebook, envie os três arquivos `.txt` para o painel de arquivos do Colab (ou monte o repositório) e use *Ambiente de execução → Executar tudo*. O `prettytable` pode precisar de `!pip install prettytable`.

**Localmente.**

```bash
git clone [URL do repositório]
cd [nome-do-repositório]
pip install numpy matplotlib pandas prettytable scikit-learn jupyter
jupyter notebook
```

Abra o notebook e execute com *Restart & Run All*. As células devem ser executadas em ordem, pois as partes seguintes reutilizam variáveis e funções anteriores. Os três arquivos `.txt` devem estar na mesma pasta do notebook.

## Reprodutibilidade

As Partes II e III usam `np.random.seed(ID)` (ID = 0), então os resultados se repetem a cada execução. Na Parte I, a semente é opcional (`SEMENTE`), e sem ela os pesos iniciais mudam a cada execução.
