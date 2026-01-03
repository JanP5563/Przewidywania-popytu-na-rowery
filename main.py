import joblib
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template
from flask import request
from flask import current_app

# Co zrobić?
# Zrób raport i przygotuj to do wysłania, posprzątaj kod

# Co mogę jeszcze zrobić?
# Dać obsługę wysłanego pustego arkusza na stronie
# Żeby strona zapisywała jakoś dane, może przekieruj na inną podstronę, idk
# Godziny tez one hot encoder


app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def main():
    if request.method=='POST':
    # Przyjmujemy dane od strony
        miesiac = int(request.form.get('miesiac'))      
        dzien = int(request.form.get('dzien'))
        godzina = int(request.form.get('godzina'))
        temperatura = float(request.form.get('temperatura'))
        temperatura_rosy = float(request.form.get('temp_rosy'))
        wilgotnosc = int(request.form.get('wilgotnosc'))
        wiatr = float(request.form.get('wiatr'))
        przejrzystosc = int(request.form.get('przejrzystosc'))
        slonce = float(request.form.get('slonce'))
        deszcz = float(request.form.get('deszcz'))
        snieg = float(request.form.get('snieg'))
        swieto = [1] if request.form.get('swieto') == "Tak" else [0] # Tutaj juz konwertujemy je na liczby, żeby było łatwiej
        praca = [1] if request.form.get('praca') == "Tak" else [0]
        pora = request.form.get('pora') # Tu są surowe

        # Zmieniamy dane, aby były dopasowane do modeli. Najpierw ogólne przekształcenia identyczne w obu modelach
        miesiac_k = [0] * 11
        if 1 <= miesiac <= 11:
            miesiac_k[miesiac-1]=1
        dzien_k = [0] * 30
        if 1 <= dzien <= 30:
            dzien_k[dzien-1]=1
        

        pora_k = [0]*3
        if pora == 'wiosna':
             pora_k[0]=1
        elif pora == 'lato':
            pora_k[1]=1
        elif pora == 'zima':
            pora_k[2]=1

        kolumny = [godzina, temperatura, wilgotnosc, wiatr, przejrzystosc, temperatura_rosy, slonce, deszcz, snieg] + swieto + praca + pora_k + dzien_k + miesiac_k 
        if 8<=godzina<=9 | 17<=godzina<=19:
             czy_szczyt = [1]
        else:
             czy_szczyt = [0]
        kolumny = kolumny + czy_szczyt

        skalar_liniowy = joblib.load('models/scaler_linear.pkl')
        model_liniowy = joblib.load('models/model_linear.pkl')

        #macierz_liniowa = np.array(kolumny).reshape(1, -1)
        nazwy_kolumn = ['Hour', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday',
       'Functioning Day', 'Seasons_Spring', 'Seasons_Summer', 'Seasons_Winter',
       'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'Day_8', 'Day_9',
       'Day_10', 'Day_11', 'Day_12', 'Day_13', 'Day_14', 'Day_15', 'Day_16',
       'Day_17', 'Day_18', 'Day_19', 'Day_20', 'Day_21', 'Day_22', 'Day_23',
       'Day_24', 'Day_25', 'Day_26', 'Day_27', 'Day_28', 'Day_29', 'Day_30',
       'Day_31', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
       'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
       'IsRushHour']
        dane = pd.DataFrame([kolumny], columns=nazwy_kolumn)

        dane_liniowe_zeskalowane = skalar_liniowy.transform(dane)
        wynik_liniowy = float(model_liniowy.predict(dane_liniowe_zeskalowane)[0])        


        # Odpalamy model xgb
        model_xgb = joblib.load('models/model_xgb.pkl')
        wynik_xgb = float(model_xgb.predict(dane)[0])

        if wynik_liniowy<0:
             wynik_liniowy=0
        if wynik_xgb<0:
             wynik_xgb=0

        wynik = f"Wynik modelu liniowego: {int(wynik_liniowy)} \n Wynik modelu XGBoost: {int(wynik_xgb)}"

        # pomoc = f"Wynik: {wynik} \n miesiac: {miesiac}, dzien: {dzien}, godzina: {godzina}, temp: {temperatura}, temp_r: {temperatura_rosy}, wilgo: {wilgotnosc}, wiatr: {wiatr}, przekrzystosc: {przejrzystosc}, slonce: {slonce}, deszcz: {deszcz}, snieg: {snieg}, swieto: {swieto}, praca: {praca}, pora roku: {pora}"
        return render_template("main.html", wynik = wynik)
    return render_template("main.html")
    

if __name__ == "__main__":
        app.run(
        "127.0.0.1",
        5001,
        debug=True
    )
        

# --- WYNIKI REGRESJI LINIOWEJ ---
# R2 Score Test: 0.4324
# (MAE) Test: 318.36 rowerów
# (MSE) Test: 184197.67 rowerów
# (RMSE) Test: 429.18 rowerów

# --- WYNIKI KOŃCOWE XGBOOST  ---
# R2 Testowy: 0.5614
# MAE Testowy: 271.2101
# MSE Testowy: 142310.8750
# RMSE Testowy: 377.2411