import streamlit as st
from openai import OpenAI
import datetime
import pickle
import pandas as pd
import numpy as np
import base64
import re
from PIL import Image
import textwrap

# Konfiguracja strony
st.set_page_config(
    layout="wide",
    page_title="AI - Ocena Ryzyka Chorób Przewlekłych",
)

# Inicjalizacja stanu sesji, aby przechowywać informację o wykonanej analizie
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
    st.session_state['results_df'] = pd.DataFrame()
    st.session_state['diagnoses_and_recommendations'] = []
    st.session_state['risk_results'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    
# Symbolic AI: definicja reguł diagnostycznych
def interpret_symptoms_and_risks(risk_results, inputs):
    """
    Zwraca listę diagnoz i rekomendacji na podstawie reguł dla pięciu chorób.
    inputs: dict z wartościami glukoza, skurcz, rozkurcz, bmi
    risk_results: dict z {'disease': {'level':..., 'prob':...}}
    """
    diagnoses = []
    glu = inputs['glukoza']
    sk = inputs['skurcz']
    roz = inputs['rozkurcz']
    bmi = inputs['bmi']
    stres = inputs.get('stres', 0)
    hdl = inputs.get('hdl', 100)

    # Cukrzyca - reguła 1: stan przedcukrzycowy
    if risk_results['Cukrzyca']['prob'] >= 0.40 and glu >= 100 and glu <= 125:
        diagnoses.append((
            "Na podstawie ryzyka wystąpienia cykrzycy oraz poziomu glukozy pacjenta, stwierdzono podejrzenie stanu przedcukrzycowego.",
            "Wizyta u diabetologa w celu wykonania testu tolerancji glukozy."
        ))
    # Cukrzyca - reguła 2: cukrzyca i kontrola HbA1c
    if risk_results['Cukrzyca']['prob'] >= 0.50 and glu > 125:
        diagnoses.append((
            "Na podstawie wysokiego ryzyka wystąpienia cykrzycy oraz podwyższonego poziomu glukozy, stwierdzono ryzyko występowania hiperglikemii i cukrzycy u pacjenta.",
            "Badanie poziomu HbA1c oraz wizyta u diabetologa lub konsultacja endokrynologiczna."
        ))

    # Nadciśnienie - reguła 3: wysokie ciśnienie
    if risk_results['Nadciśnienie']['prob'] >= 0.50 and (sk > 140 or roz > 90):
        diagnoses.append((
            "Na podstawie podwyższonego ciśnienia skurczowego i rozkurczowego, stwierdzono wysokie prawdopodobieństwo nadciśnienia.",
            "Konsultacja kardiologiczna oraz systematyczne monitorowanie ciśnienia krwi w domu."
        ))
    # Nadciśnienie - reguła 4: umiarkowane ryzyko nadciśnienia + wysoki poziom stresu
    if risk_results['Nadciśnienie']['prob'] >= 0.3 and stres > 6:
        diagnoses.append((
            "Na podstawie wyniku ryzyka nadciśnienia oraz deklarowanego poziomu stresu, stwierdzono występowanie nadciśnienia związanego ze stresem pacjenta.",
            "Techniki relaksacyjne oraz systematyczna kontrola ciśnienia krwi w domu. W przypadku braku poprawy stanu zdrowia, wizyta u internisty lub hipertensjologa."
        ))

    # Otyłość - reguła 5: otyłość kliniczna
    if risk_results['Otyłość']['prob'] >= 0.66 and bmi >= 30:
        diagnoses.append((
            "Na podstawie wysokiego ryzyka występowania otyłości oraz BMI pacjenta, stwierdzono otyłość kliniczną.",
            "Konsultacja z dietetykiem lub lekarzem bariatrą i opracowanie planu redukcji masy ciała."
        ))
    # Otyłość - reguła 6: nadwaga
    if risk_results['Otyłość']['prob'] >= 0.33 and risk_results['Otyłość']['prob'] < 0.66 and bmi >= 25:
        diagnoses.append((
            "Na podstawie ryzyka występowania otyłości oraz BMI, stwierdzono nadwagę pacjenta.",
            "Modygifikacja diety oraz zwiększenie aktywności fizycznej."
        ))

    # Choroba serca - reguła 7: wysokie ryzyko choroby serca
    if risk_results['Choroba serca']['prob'] >= 0.50 and sk > 140:
        diagnoses.append((
            "Na podstawie wyniku oceny ryzyka oraz wysokiego ciśnienia skurczowego, stwierdzono ryzyko choroby serca pacjenta.",
            "Badanie EKG oraz konsultacja u kardiologa."
        ))
    # Choroba serca - reguła 8: niski cholesterol HDL
    if risk_results['Choroba serca']['prob'] >= 0.33 and hdl < 40:
        diagnoses.append((
            "Na podstawie wyniku oceny ryzyka oraz wyników cholesterolu, stwierdzono ryzyko sercowo-naczyniowe związane z niskim poziomem HDL.",
            "Kontrola profilu lipidowego i wizyta u kardiologa."
        ))

    # Choroba nerek - reguła 9: ryzyko nerkowe przy wysokim ciśnieniu
    if risk_results['Choroba nerek']['prob'] >= 0.50 and sk > 140:
        diagnoses.append((
            "Na postawie wyniku oceny ryzyka oraz podwyższonego ciśnienia skurczowego, stwierdzono ryzyko choroby nerek u pacjenta.",
            "Badanie funkcji nerek (creatininemia, eGFR) i konsultacja nefrologiczna."
        ))
    # Choroba nerek - reguła 10: umiarkowane ryzyko i wysoki poziom glukozy
    if risk_results['Choroba nerek']['prob'] >= 0.33 and glu > 150:
        diagnoses.append((
            "Na podstawie oceny ryzyka choroby nerek oraz wysokiego poziomu glukozy, stwierdzono ryzyko występowania nefropatii cukrzycowej u pacjenta.",
            "Wizyta u diabetologa lub nefrologa oraz badanie mikroalbuminurii i ocena funkcji nerek."
        ))

    return diagnoses

# Ładowanie wytrenowanych modeli i scalera
@st.cache_resource
def load_ml_assets():
    """Ładuje model ML i scaler z plików .pkl."""
    try:
        with open('model.pkl', 'rb') as mf:
            model = pickle.load(mf)
        with open('scaler.pkl', 'rb') as sf:
            scaler = pickle.load(sf)
        return model, scaler
    except FileNotFoundError:
        st.error("Błąd: Brak plików 'model.pkl' lub 'scaler.pkl'. Upewnij się, że znajdują się w głównym folderze aplikacji.")
        return None, None

model, scaler = load_ml_assets()

# Inicjalizacja klucza API
try:
    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
except Exception as e:
    st.error(f"Błąd konfiguracji klucza OpenAI API: {e}. Sprawdź swoje sekrety w Streamlit.")
    client = None

# Górny wiersz z grafikami
st.markdown("### ")
img_row = st.columns(4)
img_paths = [
    'static/images/image1.png',
    'static/images/image2.png',
    'static/images/image3.png',
    'static/images/image4.png'
]
for col, path in zip(img_row, img_paths):
    try:
        img = Image.open(path)
        col.image(img, use_container_width=True)
    except FileNotFoundError:
        col.info(f"Brak obrazu: {path}")

# Układ dwóch kolumn poniżej grafik
left_col, right_col = st.columns([0.7, 0.3])

# Lewa kolumna - formularz i wyniki
with left_col:
    st.title('AI - Ocena Ryzyka Chorób Przewlekłych')
    with st.form('form_data'):
        st.subheader('Dane pacjenta:')
        wiek = st.number_input('Wiek:', 0, 120, 30)
        płeć = st.selectbox('Płeć:', ['Kobieta', 'Mężczyzna'])
        wzrost = st.number_input('Wzrost (cm):', 50.0, 250.0, 170.0, format="%.1f")
        waga = st.number_input('Waga (kg):', 20.0, 200.0, 60.0, format="%.1f")
        skurcz = st.number_input('Ciśnienie skurczowe:', 50, 250, 120)
        rozkurcz = st.number_input('Ciśnienie rozkurczowe:', 30, 150, 80)
        glukoza = st.number_input('Glukoza: (mg/dl):', 40.0, 300.0, 90.0, format="%.1f")
        cholesterol = st.number_input('Cholesterol całkowity (mg/dl):', 100.0, 400.0, 180.0, format="%.1f")
        hdl = st.number_input('Cholesterol HDL (mg/dl):', 10.0, 100.0, 50.0, format="%.1f")
        ldl = st.number_input('Cholesterol LDL (mg/dl):', 10.0, 300.0, 100.0, format="%.1f")
        trig = st.number_input('Triglicerydy (mg/dl):', 10.0, 500.0, 100.0, format="%.1f")
        palenie = st.selectbox('Czy palisz papierosy?', ['nie', 'tak'])
        aktywność = st.selectbox('Poziom aktywności fizycznej:', ['brak', 'umiarkowana', 'duża'])
        stres = st.slider('Subiektywny poziom stresu (0-10):', 0, 10, 5)
        przeszłość = st.selectbox('Poważna choroba w przeszłości?', ['nie', 'tak'])
        opis = st.text_area(
            "Opisz swoje samopoczucie, objawy lub stan zdrowia (opcjonalnie):",
            placeholder="Np. częste bóle głowy, osłabienie, problemy ze snem, dodatkowy stres w pracy..."
        )
        
        # Automatyczne obliczanie BMI
        if wzrost > 0 and waga > 0:
            bmi = waga / ((wzrost / 100) ** 2)
        else:
            bmi = 0.0
        
        st.number_input(
            'Wskaźnik BMI (obliczany automatycznie):',
            value=bmi,
            format="%.1f",
            disabled=True,
            help="BMI jest obliczane automatycznie na podstawie podanego wzrostu i wagi."
        )
        
        generuj = st.form_submit_button('🔍 Wykonaj analizę ryzyka')

    # Cała logika obliczeń i zapisywania do stanu sesji
    if generuj and model and scaler and client:
        with st.spinner('Analizowanie danych i generowanie wyników...'):
            # Przygotowanie cech pacjenta do predykcji
            features = np.array([
                wiek, 1 if płeć == 'Mężczyzna' else 0,
                wzrost, waga, skurcz, rozkurcz,
                glukoza, cholesterol, hdl, ldl, trig,
                1 if palenie == 'tak' else 0,
                {'brak': 0, 'umiarkowana': 1, 'duża': 2}[aktywność],
                stres, bmi,
                1 if przeszłość == 'tak' else 0
            ]).reshape(1, -1)
            
            # Skalowanie cech i predykcja
            features_scaled = scaler.transform(features)
            probabilities = model.predict_proba(features_scaled)
            
            diseases = ['Cukrzyca', 'Nadciśnienie', 'Otyłość', 'Choroba serca', 'Choroba nerek']
            risk_results = {}
            for i, disease in enumerate(diseases):
                prob = probabilities[i][0][1]
                level = 'niski' if prob < 0.33 else ('średni' if prob < 0.66 else 'wysoki')
                risk_results[disease] = {'level': level, 'prob': prob}

            # Generowanie krótkich komentarzy AI
            prompt_content = (
                "Jesteś doświadczonym lekarzem i asystentem medycznym. Na podstawie poniższych danych dotyczących ryzyka chorób oraz opisu pacjenta, "
                "wygeneruj zwięzły komentarz dla każdej z chorób, ograniczony do maksymalnie 15 słów. Bądź profesjonalny i empatyczny. "
                "Komentarz nie powinien powtarzać nazwy choroby. "
                "Oddziel komentarz dla każdej choroby unikalnym separatorem '###'.\n\n"
                f"**Opis pacjenta:** {opis if opis else 'Brak opisu.'}\n\n"
                "**Wyniki oceny ryzyka:**\n"
            )
            for disease, data in risk_results.items():
                level = data['level']
                prob_pct = f"{data['prob']:.1%}"
                prompt_content += f"- **{disease}:** Poziom ryzyka: {level} ({prob_pct})\n"

            messages = [
                {'role': 'system', 'content': "Działaj jako ekspert medyczny. Odpowiadaj precyzyjnie i zwięźle w języku polskim."},
                {'role': 'user', 'content': prompt_content}
            ]
            
            ai_comments_dict = {}
            try:
                response = client.chat.completions.create(model='gpt-4o', messages=messages)
                raw_comments = response.choices[0].message.content.strip()
                comments_list = [c.strip() for c in raw_comments.split('###') if c.strip()]
                
                if len(comments_list) == len(diseases):
                    ai_comments_dict = dict(zip(diseases, comments_list))
                else:
                    for i, disease in enumerate(diseases):
                        ai_comments_dict[disease] = comments_list[i] if i < len(comments_list) else "Błąd generowania komentarza."

            except Exception as e:
                st.error(f"Błąd podczas komunikacji z API OpenAI: {e}")
                for d in diseases:
                    ai_comments_dict[d] = "Nie udało się wygenerować komentarza z powodu błędu."

            # Przygotowanie tabeli
            table_data = []
            diseases_order = ['Cukrzyca', 'Nadciśnienie', 'Otyłość', 'Choroba serca', 'Choroba nerek']
            max_words_comment = 15

            for disease in diseases_order:
                if disease in risk_results:
                    data = risk_results[disease]
                    risk_percentage = f"{data['prob']:.1%}"
                    comment = ai_comments_dict.get(disease, 'Brak komentarza.')
                    
                    words = comment.split()
                    if len(words) > max_words_comment:
                        truncated_comment = " ".join(words[:max_words_comment]) + "..."
                    else:
                        truncated_comment = comment
                    
                    disease_name_pattern = re.compile(re.escape(disease) + r'\s*[:\.]?\s*', re.IGNORECASE)
                    truncated_comment = disease_name_pattern.sub('', truncated_comment, 1).strip()
                    truncated_comment = re.sub(r'^\*\*(.*?)\*\*\s*:\s*', r'\1', truncated_comment).strip()
                    truncated_comment = re.sub(r'^\*\*(.*?)\*\*\s*', r'\1', truncated_comment).strip()

                    table_data.append([disease, risk_percentage, truncated_comment])
            
            # Zapisywanie danych do stanu sesji
            st.session_state['results_df'] = pd.DataFrame(table_data, columns=['Choroba', 'Ocena Ryzyka (%)', 'Komentarz AI'])

            # Wywołanie i zapisanie diagnoz symbolic AI
            user_inputs = {
                'glukoza': glukoza, 'skurcz': skurcz, 'rozkurcz': rozkurcz,
                'bmi': bmi, 'stres': stres, 'hdl': hdl
            }
            st.session_state['diagnoses_and_recommendations'] = interpret_symptoms_and_risks(risk_results, user_inputs)
            
            # Zapisanie danych, aby były dostępny w chacie
            st.session_state['risk_results'] = risk_results
            
            # Flaga - analiza zakończona
            st.session_state['analysis_complete'] = True
            
    # Wyświetlanie wyników, jeśli analiza została wykonana
    if st.session_state.get('analysis_complete', False):
        st.subheader('📋 Analiza wyników badań')
        
        # Wyświetalnie tabeli (DataFrame)
        st.dataframe(
            st.session_state['results_df'],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Choroba": st.column_config.Column(width=80),
                "Ocena Ryzyka (%)": st.column_config.Column(width=80),
                "Komentarz AI": st.column_config.Column(width=600)
            }
        )

        # Wyświetlenie diagnoz i rekomendacji
        diagnoses_and_recommendations = st.session_state.get('diagnoses_and_recommendations', [])
        if diagnoses_and_recommendations:
            st.markdown("---")
            st.subheader('🩺 Dodatkowa diagnoza i rekomendacje')
            for i, (diagnosis, recommendation) in enumerate(diagnoses_and_recommendations):
                st.markdown(f"**Diagnoza:** {diagnosis}")
                st.markdown(f"**Rekomendacja:** {recommendation}")
                if i < len(diagnoses_and_recommendations) - 1:
                    st.markdown("---")
        else:
            st.info("Brak dodatkowych spostrzeżeń diagnostycznych na podstawie bieżących danych.")

        # Zawsze wyświetlaj zastrzeżenie po wynikach
        st.markdown('---')
        st.subheader('⚠️ Zastrzeżenie')
        st.markdown("""
            Wyniki prezentowane przez aplikację mają wyłącznie charakter informacyjny i edukacyjny. Nie stanowią one diagnozy medycznej, porady lekarskiej ani nie mogą być traktowane jako substytut konsultacji z wykwalifikowanym pracownikiem służby zdrowia. Algorytmy i modele wykorzystywane w aplikacji opierają się na ogólnych danych i nie uwzględniają wszystkich indywidualnych czynników zdrowotnych użytkownika.
        
            Zaleca się, aby wszelkie decyzje dotyczące zdrowia podejmować wyłącznie po konsultacji z lekarzem lub innym specjalistą medycznym. W przypadku jakichkolwiek wątpliwości dotyczących stanu zdrowia, objawów chorobowych lub wyników uzyskanych w aplikacji – skontaktuj się z odpowiednim specjalistą.

            Autor aplikacji nie ponosi odpowiedzialności za jakiekolwiek decyzje podjęte na podstawie prezentowanych wyników ani za ich konsekwencje.
            """)
                    

# Prawa kolumna - opis aplikacji
with right_col:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### Opis Aplikacji")
    st.markdown(
    """
    <div style="text-align: justify;">

    **AI Ocena Ryzyka Chorób Przewlekłych** to prosty agent AI, który łączy w sobie kilka kluczowych dziedzin sztucznej inteligencji, tworząc platformę diagnostyczno-rekomendacyjną dla wyników badań użytkownika.
    Agent bazuje na sztucznej inteligencji, wykorzystując:

    * przetwarzanie języka naturalnego (NLP)
    * model uczenia maszynowego (ML)
    * duży model językowy (LLM)
    * Symbolic AI
    * Streamlit
    * biblioteki python (m.in pickle i pandas)

    Aplikacja ma proste w obsłudze działanie - po wprowadzeniu przez użytkownika swoich danych oraz wyników badań, generowane zostają: analiza wyników badań oraz dodatkowe diagnozy i rekomendacje modelu. 
    Opisywany agent AI używa ML do budowy algorytmu i trenowania modelu na podstawie wygenerowanych danych fikcyjnych pacjentów. Model analizuje dane oraz zwraca prawdopodobieństwo wystąpienia pięciu chorób przewlekłych.
    W aplikacji wykorzystano również NLP oraz LLM w celu udostępnienia uytkownikowi możliwości wprowadzenia subiektywnego opisu samopoczucia oraz obecnego stanu zdrowia. Tekst jest uwzględniany w promptach wysyłanych do modelu GPT‑4o, który generuje wynik na podstawie wszystkich zebranych danych.

    Kolejnym użytym rozwiązaniem jest Symbolic AI, które zostało zainicjowane w kodzie jako zestaw reguł dla dodatkowych diagnoz i rekomendacji. 
    Reguły eksperckie zostały opracowane na podstawie powszechnie akceptowanych wytycznych medycznych. Dla każdej z pięciu chorób zdefiniowano dwie reguły logiczne, które łączą wynik ML z wartościami bezwzględnymi (np. ciśnienie, BMI). 
    System analizuje te reguły i w przypadku spełnienia warunków – generuje konkretne diagnozy i rekomendacje.

    Całość procesu jest osadzona w Streamlit, co pozwala na wyświetlenie aplikacji w nowoczesnym interfejsie. Wykorzystująć go, użytkownik może w szybki i prosty uzupełnić wszystkie potrzebne dane. 
    Po ich zatwierdzeniu, użytkownik otrzymuje w jednej, skonsolidowanej tabeli: ocenę ryzyka oraz kró†ki komentarz AI.
    Poniżej tabeli mogą pojawić się wcześniej wspomniane dodatkowe diagnozy i rekomendacje (tylko w przypadku spełnienia założeń reguły).

    Ostatnim narzędziem zastosowanym w tej aplikacji medyczny chatbot, gdzie użytkownik może zapytać o kolejne i bardziej dokładne informacje dotyczące np. otrzymanego wyniku i diagnozy.
    Stworzony agent AI jest gotowy na dalszy rozwój. Architektura pozwala na łatwe dodanie kolejnych modułów AI: np. analiza obrazów (CNN), pamięć z historią pacjtena czy też rekomendacja konkretnych specjalistów dla danych miejscowości. 

    Stworzona aplikacja jest bezpieczna — dane użytkownika nie są zapisywane, klucz API OpenAI nie jest przechowywany w repozytorium, a aplikację można uruchamiać zarówno lokalnie, jak i online.

    Kamil Lemański 2025©
    </div>
    """,
    unsafe_allow_html=True
    )

    # Chatbot medyczny
    from streamlit.delta_generator import DeltaGenerator
    st.markdown("### 💬 Medyczny Chatbot")
    st.info("Zadaj pytanie dotyczące Twoich wyników lub rekomendacji. Asystent odpowiada wyłącznie w kontekście danych medycznych.")
    
    # Inicjalizacja historii czatu
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Historia rozmowy z chatbotem
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.chat_message('user').write(msg['content'])
        else:
            st.chat_message('assistant').write(msg['content'])

    # Wejście użytkownika
    user_question = st.chat_input("Masz pytanie na temat wyników? Wpisz je w tym oknie.")
    if user_question:
        # Dodaj wiadomość użytkownika do historii
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})

        # Prompt dla agenta (chatbot)
        system_prompt = (
            "Jesteś profesjonalnym asystentem medycznym. "
            "Masz dostęp do wyników ML i rekomendacji symbolicznych pacjenta. "
            "Odpowiadaj tylko na tematy medyczne, wykorzystaj kontekst danych:"
        )
        # Pobranie danych z sesji
        sr = st.session_state.get('risk_results', {})
        dr = st.session_state.get('diagnoses_and_recommendations', [])
        context = f"""Wyniki ML: {sr}
Rekomendacje: {dr}"""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'assistant', 'content': context},
            {'role': 'user', 'content': user_question}
        ]

        # Wywołanie klucza API
        try:
            response = client.chat.completions.create(model='gpt-4o', messages=messages)
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = "Przepraszam, wystąpił błąd przy przetwarzaniu Twojego pytania."
            st.error(f"Błąd API OpenAI: {e}")

        # Dodaj odpowiedź do historii i wyświetl
        st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
        st.chat_message('assistant').write(answer)