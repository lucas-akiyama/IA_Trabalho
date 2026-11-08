# Análise Comparativa de Algoritmos de Classificação: MLP vs. Random Forest

**Disciplina:** Inteligência Artificial (UFABC)
**Atividade:** Trabalho 1, em grupo
**Integrantes:** Gustavo dos Santos Moura, Lucas Akiyama de Moura, Pedro Augusto Sanches Pires

Previsão do time vencedor em partidas de dois MOBAs (classificação binária),
comparando uma rede neural (`MLPClassifier`) e um `RandomForestClassifier`.

## Bases de dados

| Base | Fonte | Tamanho | Alvo |
|---|---|---|---|
| Dota 2 Games Results | UCI Machine Learning Repository | 92.650 treino / 10.294 teste (divisão oficial) | Radiant venceu = 1 |
| League of Legends Ranked Games (`games.csv`) | Kaggle | 51.490 partidas (divisão 80/20 feita no código) | Time 1 venceu = 1 |

Cenários experimentais:

- `dota2`: draft de heróis (+ região, modo e tipo de jogo);
- `lol_draft_only`: draft, bans e feitiços (somente informação pré-partida);
- `lol_draft_ingame`: adiciona estatísticas da partida. É um **contraste
  deliberado de vazamento de dados**, não um preditor válido.

## Requisitos

- Python 3.10 ou superior
- Apenas as bibliotecas permitidas no enunciado: `numpy`, `pandas`,
  `scikit-learn`, `matplotlib` e `seaborn` (ver `requirements.txt`)
- `make` (opcional; todos os comandos têm equivalente em Python puro)

## Instalação

```bash
make install
```

ou, manualmente:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Git Bash/Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

## Como carregar as bases de dados

Os dados **não** são versionados (arquivos grandes). Baixe-os e coloque-os
exatamente nestes caminhos, a partir da raiz do projeto:

```
data/raw/dota2/dota2Train.csv
data/raw/dota2/dota2Test.csv
data/raw/lol/games.csv
```

1. **Dota 2:** página *Dota2 Games Results* no UCI Machine Learning Repository.
   Extraia o `.zip` e copie `dota2Train.csv` e `dota2Test.csv` (os arquivos
   `.json` auxiliares não são usados). Os arquivos não têm cabeçalho.
2. **LoL:** dataset *League of Legends Ranked Games* no Kaggle. Copie o
   `games.csv`.

Verificação e teste de carregamento:

```bash
make check-data
make smoke            # python -m src.data_loading
```

## Como executar os experimentos

Todos os comandos são executados na raiz do projeto, com o ambiente ativo.

| Objetivo | Com make | Sem make |
|---|---|---|
| Verificação rápida (grades reduzidas) | `make quick` | `python -m src.main --quick` |
| Experimento completo (3 cenários) | `make run` | `python -m src.main` |
| Apenas Dota 2 | `make dota2` | `python -m src.main --scenarios dota2` |
| Apenas LoL | `make lol` | `python -m src.main --scenarios lol_draft_only lol_draft_ingame` |
| Testes de significância (McNemar) | `make significance` | `python -m src.significance` |
| Reconstruir `summary.csv` e gráfico | `make summary` | `python -c "from src import main; main.combine_summaries()"` |
| Verificar bibliotecas proibidas | `make lint` | (ver `Makefile`) |

> **Atenção:** `make quick` sobrescreve os arquivos em `results/`. Rode
> `make run` depois dele para regenerar os resultados finais, e
> `make significance` por último (ele lê os melhores hiperparâmetros de
> `results/tables/*_summary.csv`).

Tempo aproximado (CPU comum): Dota 2 ≈ 6 min; LoL `draft_only` ≈ 3 min;
`draft_ingame` ≈ 4 min; testes de significância ≈ 3 a 5 min.

## Saídas

- `results/tables/`: `<cenário>_<modelo>_tuning.csv` (busca de hiperparâmetros),
  `<cenário>_<modelo>_seeds.csv` (métricas por semente),
  `<cenário>_<modelo>_confusion.csv`, `<cenário>_summary.csv`, `summary.csv`
  e `significance.csv`;
- `results/figures/`: matrizes de confusão, gráficos de hiperparâmetros e
  `model_comparison.png`.

## Estrutura do projeto

```
src/
  config.py          caminhos, sementes, grades de hiperparâmetros
  data_loading.py    leitura das bases e padronização do alvo
  preprocessing.py   pré-processamento sem data leakage, codificação de presença
  models.py          construtores do MLP e do RF, busca de hiperparâmetros
  evaluation.py      métricas, matrizes de confusão, baseline
  plots.py           gráficos de hiperparâmetros e comparação
  main.py            pipeline completo (3 cenários)
  significance.py    teste de McNemar e bootstrap pareado
results/             tabelas e figuras geradas
report/              relatório técnico (PDF)
```

## Reprodutibilidade

- Sementes fixas em `src/config.py` (`RANDOM_STATE = 42`, `SEEDS`).
- Versões das bibliotecas fixadas em `requirements.txt` (`make freeze`).
- Os resultados podem variar levemente entre versões do scikit-learn e entre
  sistemas operacionais.

## Restrições do enunciado

Somente `numpy`, `pandas`, `scikit-learn`, `matplotlib` e `seaborn`.
O comando `make lint` falha se encontrar TensorFlow, Keras, PyTorch, XGBoost,
LightGBM, CatBoost ou similares em `src/`.
