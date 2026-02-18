projeto-ml-sota/
│
├── .devcontainer/       # Config do Docker (ja tem)
├── .dvc/                # Config interna do DVC (não mexa aqui)
├── .github/             # Pipelines de CI/CD (futuro)
│
├── conf/                # HYDRA: Configurações dos experimentos
│   ├── base/
│   │   ├── train.yaml   # Hiperparâmetros (lr, batch_size)
│   │   └── data.yaml    # Caminhos dos dados
│   └── config.yaml      # Arquivo principal
│
├── data/                # DVC: Onde os dados MORAM (Ignorado pelo Git)
│   ├── raw/             # Dados brutos (X.csv, y.csv)
│   ├── processed/       # Dados limpos/features
│   └── .gitignore       # O DVC cria isso para impedir o Git de ver os CSVs
│
├── models/              # DVC: Onde os modelos salvos (.pkl) ficam
│
├── notebooks/           # EDA e Rascunhos (Git)
│
├── reports/             # Artefatos Finais
│   ├── figures/         # Gráficos gerados (PNG/PDF)
│   └── paper/           # Código LaTeX do Beamer
│       └── main.tex
│
├── src/                 # SEU CÓDIGO PYTHON (Git)
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py  # Script ETL/Cleaning
│   ├── features/
│   │   └── build_features.py # Feature Engineering
│   ├── models/
│   │   └── train_model.py    # Treino (Hydra + MLflow)
│   └── visualization/
│       └── plot_results.py   # Gera gráficos para o Beamer
│
├── dvc.yaml             # O ORQUESTRADOR (Git)
├── dvc.lock             # O ESTADO CONGELADO (Git)
├── poetry.lock          # Dependências Python (Git)
├── pyproject.toml       # Config do Projeto (Git)
└── README.md