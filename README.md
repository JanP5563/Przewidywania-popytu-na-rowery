# Przewidywanie popytu na rowery w Seulu

Projekt realizowany w ramach studiów. 
Aplikacja przewiduje liczbę wypożyczanych rowerów na podstawie danych pogodowych i czasowych.

## Jak uruchomić projekt
1. Sklonuj repozytorium.
2. Stwórz wirtualne środowisko: `python -m venv venv`.
3. Zainstaluj biblioteki: `pip install -r requirements.txt`.
4. Uruchom aplikację: `python main.py`.

## Wykorzystane modele
W projekcie porównujemy dwa podejścia:
* **Model Prosty:** Regresja Liniowa.
* **Model Złożony:** XGBoost.

## Technologie
* **Backend:** Flask
* **AI/ML:** Scikit-learn, Pandas, Google Colab (trening)
* **Frontend:** HTML5, CSS

## Dane
**Dataset**: Seoul Bike Sharing Demand, UCI Machine Learning Repository

**Licencja**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Źródło**: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand