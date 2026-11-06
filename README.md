# Análise Comparativa de Algoritmos de Classificação: MLP vc. Random Forest

**Disciplina:** Inteligência Artificial (UFABC)

## Base de dados
1. **Dota 2 Games Results** (UCI Machine Learning Repository)
2. **League of Legends Ranked Games** (`games.csv`, Kaggle)

## Algoritmos
- `MLPClassifier` (scikit-learn)
- `RandomForestClassifier` (scikit-learn)

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados
Os dados NÃO são versionados. Baixe-os e coloque-os em:
- `data/raw/dota2/` (arquivos do UCI)
- `data/raw/lol/games.csv` (Kaggle)

## Execução
```bash
python -m src.main --quick                 # verificação rápida (grades reduzidas)
python -m src.main                         # experimento completo (3 cenários)
python -m src.main --scenarios dota2       # apenas um cenário
```
Saídas: `results/tables/` (CSVs) e `results/figures/` (PNGs).

## Estrutura
- `src/`: código-fonte
- `results/`: figuras e tabelas geradas
- `report/`: relatório técnico
