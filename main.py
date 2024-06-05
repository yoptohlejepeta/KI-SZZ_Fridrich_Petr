import numpy as np
import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, ttest_ind, shapiro


st.title('SZZ')

st.write(f'Zadání práce:')
zad = """
Cílem úlohy je analyzovat data z přiloženého souboru s výsledky studentů v přípravných kurzech matematiky a 
identifikovat vzájemné vztahy a souvislosti mezi různými proměnnými. Úkolem je provést explorační analýzu 
dat, formulovat statistické hypotézy a testovat je pomocí inferenčních statistických metod. Výstupem bude 
**komentovaný report** obsahující statistickou analýzu vstupních dat a interaktivní dashboard pro vizualizaci dat.


1. Načtení a příprava dat: Načtěte data do prostředí R, zkontrolujte kvalitu dat a proveďte potřebné úpravy, jako je například odstranění chybějících hodnot a duplicit nebo převedení proměnných do vhodných datových typů.
2. Explorační analýza dat: Charakterizujte jednotlivé proměnné a prozkoumejte vztahy mezi nimi. Použijte metody popisné statistiky a vizualizační techniky, jako jsou histogramy, boxploty, scatterploty a heatmapy.
3. Formulace statistických hypotéz: Na základě zjištěných vztahů a skupin navrhněte statistické hypotézy, které budou testovány pomocí inferenčních statistických metod.
4. Testování statistických hypotéz: Použijte metody inferenční statistiky, jako jsou intervaly spolehlivosti, regresní analýza, analýza rozptylu (ANOVA), nebo neparametrické testy, pro testování navržených hypotéz a určení statistické významnosti zjištěných vztahů.
5. Vytvoření interaktivního dashboardu: Vytvořte interaktivní dashboard pomocí knihoven Shiny a shinydashboard, který umožní uživatelům procházet výsledky analýzy a vybírat, co a v jaké podobě chtějí vidět. Dashboard by měl zahrnovat grafy, tabulky a filtry pro různé proměnné či jejich kombinace.
6. Komentovaný report: Prezentujte výsledky statistické analýzy a vizualizace ve formě komentovaného reportu, který obsahuje výstupy z jednotlivých kroků, interpretaci výsledků a závěry. Report vytvořte ve formátu R Markdown či Quarto.

Použité technologie mohou být jiné než zmiňované R (tidyverse), Quarto, Shiny a shinydashboard. Preferovanou další volbou je Python (Pandas, NumPy, Matplotlib/Seaborn/ Plotly, Scipy, Statsmodels), Jupyter Notebook, Dash/Streamlit.
"""

st.write(zad)

# ============================================================================= #

data = pd.read_csv('Data/Math.csv', sep=';')

st.title("První pohled na data")

default_columns = data.columns.tolist()[:5]

selected_columns = st.multiselect(
    "Zde si můžete vybírat sloupce, které si chcete zobrazit",
    options=data.columns.tolist(),
    default=default_columns
)

if selected_columns:
    st.dataframe(data[selected_columns])
else:
    st.write("Please select columns to display.")

st.title('Informace o datech a čištění')

st.write(f'Následující výpis ukazuje, že máme 16 sloupců. Každý sloupec má 83 non-null hodnot. Z velké části je tvořen datovým typem Int64. Sloupec Student je indexem. Ve výpisu lze vidět i využití paměti.')

buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

data = data.set_index('Student')

data_spojita = data[['PSATM', 'SATM', 'ACTM', 'Rank', 'Size', 'GPAadj', 'PlcmtScore']]


# ============================================================================= #

st.write(f'Následující tabulka ukazuje výpis kolik je chybějících hodnot v daném sloupci.')
nadata = data.isna().sum()

st.write(nadata)
st.write(f'Počet duplicitních řádků v datech je: {np.sum(data.duplicated())}')

# ============================================================================= #

st.write(f'Tabulka níže popisuje hodnoty boxplotů sloupců, u kterých má boxplot smysl vytvářet. Např. u sloupce Course není moc platné ho vytvářet, je to pouze identifikátor kurzu. Boxploty můžete naleznout pod tabulkou.')

txt = '''
    První řádek reprezentuje průměrné hodnoty +- odchylka, která je v řádku druhém.
    Dále jsou zde popsány jednotlivé části, hodnoty boxplotů minimum Q1, Q2, Q3, maximum.

    Jsou zde pouze hodnoty, ze kterých je vhodné vytvářet boxplot. Například u sloupce Course nemá boxplot smysl, protože se jedná se o identifikátor kurzu.
'''
st.write(data_spojita.describe())

fig, ax = plt.subplots()
sns.boxplot(data=data_spojita, ax=ax)
plt.xticks(rotation = 90)
st.pyplot(fig)

text = '''
V rámci této části jsem zkoumal jaké datové typy se v datech vyskytují. Zejména to jsou celá čísla, dále je tu několik řetězcových hodnot. 
Dále v datech nejsou chybějící ani duplicitní hodnoty.
'''

text = '''
V práci dále jsem převedl sloupce, které obsahovaly řetězce na číselné reprezentace. Pair plot, který ukazuje vztahy mezi číselnými hodnotami nyní zahrnuje i tyto sloupce.
'''

st.write(text)

# ============================================================================= #
# Zde převádím řetězce data na číselné reprezentace

grade_mapping = {
    'A+': 1,
    'A': 2,
    'A-': 3,
    'B+': 4,
    'B': 5,
    'B-': 6,
    'C+': 7,
    'C': 8,
    'C-': 9,
    'D+': 10,
    'D': 11,
    'D-': 12,
    'E+': 13,
    'E': 14,
    'E-': 15
}

from sklearn.preprocessing import LabelEncoder # lze použít pandas.get_dummies, je to dosti podobné jen to vytváří sloupečky navíc s hotnotami 0,1
label_encoder_Gender = LabelEncoder()
label_encoder_Recommends = LabelEncoder()
# label_encoder_Grade = LabelEncoder()

data['Gender'] = label_encoder_Gender.fit_transform(data['Gender'])
data['Recommends'] = label_encoder_Recommends.fit_transform(data['Recommends'])
# data['Grade'] = label_encoder_Grade.fit_transform(data['Grade'])

data['Grade'] = data['Grade'].map(grade_mapping)

# ============================================================================= #

# Pomocné funkce pro vizualizaci grafů
def plot_pairplot(dataframe):
    fig = sns.pairplot(dataframe, hue='Gender')
    plt.xticks(rotation = 90)
    return fig

def plot_violinplot(dataframe):
    fig, ax = plt.subplots()
    sns.violinplot(data=dataframe, ax=ax)
    plt.xticks(rotation = 90)
    return fig

def plot_corr(dataframe):
    correlation_matrix = dataframe.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', vmin=-1.0, vmax=1.0) 
    plt.xticks(rotation = 90)

    for t in ax.texts:
        if float(t.get_text())>=0.6 or float(t.get_text())<= -0.6 :
            t.set_text(t.get_text())
        else:
            t.set_text("")
    return fig

def plot_histogram(dataframe):
    fig, ax = plt.subplots(figsize=(10, 6))
    dataframe.hist(ax=ax, bins=10)
    fig.tight_layout()
    return fig

def plot_boxplot(dataframe):
    fig, ax = plt.subplots()
    sns.boxplot(data=dataframe, ax=ax)
    plt.xticks(rotation = 90)
    return fig

# ============================================================================= #

st.title(f'Vizualizace dat')

plot_type = st.selectbox(
    "Vyberte typ grafu k zobrazení",
    ( "Violin Plot", "Korelace", "Pair Plot", "Histogram")
)

if plot_type == "Violin Plot":
    st.pyplot(plot_violinplot(data))

elif plot_type == "Korelace":
    st.pyplot(plot_corr(data_spojita))

elif plot_type == "Pair Plot":
    st.pyplot(plot_pairplot(data))

elif plot_type == "Histogram":
    st.pyplot(plot_histogram(data))

st.write(f'V této fázi jsou data plotnutá ruznými způsoby. Z histogramů lze odhadnout rozdělení atd. Ovšem z violin plotů toho nelze moc pozorovat proto níže je stejný pohled na data jen jsou škálována na rozmezí 0,1')

st.markdown("___")

# ============================================================================= #

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=list(data.columns))

scaler_spojite = MinMaxScaler()
data_spojite_scaled = scaler.fit_transform(data_spojita)
data_spojite_scaled = pd.DataFrame(data_spojite_scaled, columns=list(data_spojita.columns))

plot_type = st.selectbox(
    "Vyberte typ grafu k zobrazení",
    ( "Violin Plot", "Korelace", "Pair Plot", "Histogram", "Box Plot")
)

if plot_type == "Violin Plot":
    st.pyplot(plot_violinplot(data_scaled))
    txt = '''
        Violin plot kombinuje informace z boxplotů a hustotních grafů. Tvar violy ukazuje hustotu dat. širší části naznačují vyšší hustotu (více datových bodů v daném intervalu.
        Pokud ve violin plotu vidíme vlnky (více peaků) může to indikovat, že v datech je 2 či více (podle počtu peaků) hlavních skupin. Například u plotu Gender jsou dva peaky to indikuje dvě pohlaví.

        Nebo u Recommends máme 3 peaky z toho můžeme posoudit, že se doporučují zejména 3 úrovně kurzů.
    '''    
    st.write(txt)

elif plot_type == "Korelace":
    st.pyplot(plot_corr(data_spojite_scaled))
    txt = '''
        Zde je zobrazena heatmapa korelací. Hodnoty na diagonále jsou 1. Dále je matice symetrická tzn A^T = A.
        
        Anotace (hodnoty) jsou zobrazeny tam, kde je korelace větší než 6 nebo menší než -6.
        Na základě tohoto odkazu: https://statisticsbyjim.com/basics/correlations/

        Zde vidět korelace jednotlivých testů PSATM, SATM, ACTM pokud byl student dobrý v jednom byl dobrý i v ostatních.
        Je také výrazná záporná korelace mezi sloupcem Rank a GPAadj.
        Kladná korelace mezi PlcmtScore a ACTM je vyšší než mezi PlcmtScore a PSATM nebo SATM to znamená, že PlcmtScore více záviselo na tom jak se studentovi vedlo v ACTM testu.
    '''    
    st.write(txt)

elif plot_type == "Pair Plot":
    st.pyplot(plot_pairplot(data_scaled))

    txt = '''
        Pairplot zobrazuje dvojice atributů datasetu proti sobě ve formě rozptylových grafů (scatter plotů), 
        a také hustotní grafy (density plots) na diagonále. Hlavní účel pairplotu je umožnit rychlou 
        vizuální inspekci vztahů mezi všemi páry číselných proměnných v datasetu.

        Bohužel nýnější vizualizace není moc vidět proto je níže scatter plot pro jednotlivé dvojice, které zadáte. Jde o přiblíženější pohled na scatter ploty.
    '''
    st.write(txt)

elif plot_type == "Histogram":
    st.pyplot(plot_histogram(data_scaled))

    txt = """
        Pomáhá vizualizovat frekvence výskytu hodnot v datasetu rozdělením dat do několika binsů. 
        Histogram je skvělý nástroj pro identifikaci tvaru rozložení, jako jsou 
        normální rozložení, rovnoměrné atd.
    """
    
    st.write(txt)

elif plot_type == "Box Plot":
    st.pyplot(plot_boxplot(data_spojite_scaled))

    txt = r"""
        Boxplot poskytuje přehled o variabilitě a symetrii dat, stejně jako o případných odlehlých hodnotách.
        1. Krabice (box): Centrální část boxplotu, která sahá od dolního kvartilu (Q1) po horní kvartil (Q3). Krabice tedy představuje středních 50 % dat.
        2. Medián (čára uvnitř krabice): Čára uvnitř krabice označuje medián dat (Q2).
        3. "Fousy" (whiskers): Čáry, které se táhnou z krabice k minimální a maximální hodnotě dat, pokud nejsou odlehlé. Obvykle se definují jako 1,5násobek interkvartilního rozpětí (IQR) od Q1 a Q3.
        4. Odlehlé hodnoty (outliers): Jednotlivé body, které leží mimo rozsah fousů. Tyto hodnoty jsou často vizualizovány jako jednotlivé tečky a indikují odlehlé hodnoty.
    """    
    st.write(txt)

st.markdown("___")

st.write(f'Zde je způsob jak vizualizovat jednotlivé atributy proti sobě, každou kombinaci lze vidět v Pair Plotu toto slouží jen pro bližší pohled. Barvička znázorňuje Gender')

def plot_scatter(dataframe, x_col, y_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataframe[x_col], dataframe[y_col], c=data_scaled['Gender'], cmap='viridis')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Scatter Plot {x_col} vs {y_col}')
    return fig

st.write()
columns = data_scaled.columns.tolist()
x_col = st.selectbox("Vyberte sloupec na X-ovou osu", columns)
y_col = st.selectbox("Vyberte sloupec na Y-ovou osu", columns)

st.pyplot(plot_scatter(data_scaled, x_col, y_col))

# ============================================================================= #

st.title('Hypotézy')

# https://statsandr.com/blog/what-statistical-test-should-i-do/images/overview-statistical-tests-statsandr.png
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/


st.write('### 1. Hypotéza')

hyp = """
Vytvořil jsem hypotézu, kdy velikost třídy (počet studentů) ovlivňuje výkonnost studentů.
Myšlenka byla taková, jestli velké množství studentů neznervozní ostatní.


+ Nulová hypotéza (H0): Velikost třídy (Size) nemá vliv na výsledky studentů (GPAadj).
+ Alternativní hypotéza (H1): Velikost třídy (Size) má vliv na výsledky studentů (GPAadj).

Použil jsem Pearsonův korelační test.
"""
st.write(hyp)

corr, p_value = pearsonr(data_scaled['Size'], data_scaled['GPAadj'])

st.write('**Výsledky Pearsonova korelačního testu**')
st.write(f'Pearsonův korelační koeficient: {corr}')
st.write(f'p-hodnota: {p_value}')

st.write('**Interpretace**')
if p_value < 0.05:
    st.write('Na základě p-hodnoty je výsledek statisticky významný.')
else:
    st.write('Na základě p-hodnoty není výsledek statisticky významný.')

st.write(f'**Závěr**')

zav = """
Na základě Pearsonova korelačního testu můžeme konstatovat, že existuje slabý, ale statisticky významný negativní vztah mezi velikostí třídy a akademickým výkonem studentů.
P-hodnota < 0.05 to znamená, že mám dostatek důkazů a zamítám H0.
"""
st.write(zav)

st.write('### 2. Hypotéza')

hyp = """
Vytvořil jsem hypotézu vlivu pohlaví na průměrné skóre z testů matematiky.


+ Nulová hypotéza (H0): Neexistuje žádný rozdíl v GPAadj mezi mužskými a ženskými studenty.
+ Alternativní hypotéza (H1): Existuje rozdíl v  GPAadj mezi mužskými a ženskými studenty.

Použil jsem dvouvýběrový t-test.
"""
st.write(hyp)

male_gpa = data_scaled[data_scaled['Gender'] == 0]['GPAadj']
female_gpa = data_scaled[data_scaled['Gender'] == 1]['GPAadj']

t_stat, p_value_gender = ttest_ind(male_gpa, female_gpa, equal_var=False)

st.write('**Výsledky t-testu**')
st.write(f't-statistika: {t_stat}')
st.write(f'p-hodnota: {p_value_gender}')

st.write('**Interpretace**')
if p_value_gender < 0.05:
    st.write('Na základě p-hodnoty s hladinou významnosti 0.05 je výsledek statisticky významný.')
else:
    st.write('Na základě p-hodnoty s hladinou významnosti 0.05 není výsledek statisticky významný.')

st.write(f'**Závěr**')

zav = """
Na základě výsledků z dvouvýběrového t-testu je p-hodnota větší než 0.05 proto nezamítám H0.
"""
st.write(zav)

st.write('### 3. Hypotéza')

hyp = """
Testování zda spojité proměnné mají normální rozdělení pomocí Shapiro-Wilk test.

+ Nulová hypotéza (H0): Data pocházejí z normálního rozdělení.
+ Alternativní hypotéza (H1): Data nepocházejí z normálního rozdělení.

"""
st.write(hyp)

cols = list(data_spojite_scaled.columns)

column_to_test = st.selectbox(
    "Vyberte sloupec, který chcete testovat", cols
)

stat, p_value = shapiro(data_spojite_scaled[column_to_test])

st.write(f'**Výsledky Shapiro-Wilk testu pro sloupeček {column_to_test}**')
st.write(f'Hodnota testu: {stat}')
st.write(f'p-hodnota: {p_value}')

st.write('**Interpretace na základě p-hodnoty**')
if p_value < 0.05:
    st.write('Zamítám H0')
else:
    st.write('Nezamítám H0')


if __name__ == '__main__':

    print('Hello!')