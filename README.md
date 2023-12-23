----------------
# MLops_projet_2023
Répertoires contenant les développements effectués dans le cadre de notre projet MLops au sein du Master SIAD
----------------
Demorgny Baptiste & Lacroix Maxime
M2 SIAD
----------------
Ce projet vise à prédire la vaccination des patients concernant la grippe saisonière mais aussi le virus H1N1.
----------------
Démarrer l'environnement virtuel '.venv'
- Se placer dans le dossier MlOps sur un terminal
- éxécuter la commande pour démarrer l'environnement de développement : .venv\Scripts\activate
----------------
Installation des dépendance
- éxécuter la commande : pip install -r requirements.txt
----------------
Exécution du projet
- Le projet correspond au fichier Clean_code.py
- éxécuter le projet avec la commande : python .\Clean_code.py
----------------
MLFlow
- après avoir éxécuté Clean_code.py, éxécuter dasn le terminal : mlflow ui
- cliquer sur l'adresse indiqué dans le terminal
- Ctrl + c pour stopper le processus
----------------
Pytest
- après avoir éxécuté Clean_code.py, éxécuter dasn le terminal : pytest pytest_Random_forest_func.py
