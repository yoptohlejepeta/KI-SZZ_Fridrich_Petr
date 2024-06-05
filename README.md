Základní popis 
1. Řešení jsem vytvářel v prostředí VScode.
2. Pomocí poetry jsem si vytvořil virtuální prostředí. 
4. Nastavil jsem interpreter z virtuálního prostředí. (CTRL + Shift + P) vyhledám Select Interpreter (enter) a zvolím ten z virtálního prostředí.
3. Dále jsem nainstaloval potřebné balíčky.

struktura:
+ složka Data obsahuje data
+ složka .venv je virtuální prostředí
+ main.py je script, který obsahuje veškeré zdrojové kódy
+ poetry.lock a pyproject.toml jsou soubory vytveřené poetry

Pro spuštění streamlitu: **poetry run streamlit run main.py**

Případně:

+ Pro spuštění je důležité mít poetry zde je odkaz na instalaci: https://pypi.org/project/poetry/ 

+ Přepnout interpreter na virtuální, který vytvořilo poetry.
