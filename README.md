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
python -m src.main
```

## Estrutura
- `src/`: código-fonte
- `notebooks/`: análise exploratória
- `results/`: figuras e tabelas geradas
- `report/`: relatório técnico
