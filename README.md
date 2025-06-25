AI - Ocena Ryzyka Chorób Przewlekłych

Live App ➤ https://ai-agent-doctor-klemanski.streamlit.app

AI Ocena Ryzyka Chorób Przewlekłych to prosty agent AI, który łączy w sobie kilka kluczowych dziedzin sztucznej inteligencji, tworząc platformę diagnostyczno-rekomendacyjną dla wyników badań użytkownika. 

------------
✨ Właściwości:

🔧 Hybrydowa anliza ryzyka (ocena ryzyka 5 chorób przewlekłych)

💬 Spersonalizowane, krótkie komentarze AI

🩺 Dodatkowe diagnozy i rekomendacje (aplikacja generuje konkretne diagnozy i zalecenia)

🤖 Medyczny chatbot kontekstowy (odpowiada w kontekście wyników badań i wygenerowanej analizy)

📱 Nowoczesny i responsywny interfejs Streamlit

🔒 Bezpieczeństwo i prywatność danych (dane użytkownika nie są zapisywane, klucz API OpenAI jest bezpiecznie przechowywany)

-------------
🧪 Zastosowane technologie:

Python 3.9+

Streamlit

OpenAI GPT-4o

ML (Scikit-learn)

Symbolic AI

Pandas

Pickle

-------------
📊 Macierz korelacji:

Na podstawie wygenerowanych danych wyjściowej oraz skalowania cech, stworzono macierz korelacji przestawiającą zależności pomiędzy danymi w aplikacji.
Przygotowana macierz prezentuje wartości współczynnika Pearsona pomiędzy 21 zmiennymi demograficznymi, antropometrycznymi, biochemicznymi, czynnikami stylu życia oraz diagnozami chorobowymi w badanym zbiorze.

Grafika przedstawiająca macierz: https://github.com/KamilLemanski/ai-agent-doctor/blob/main/corr.png

Analiza i wnioski: https://github.com/KamilLemanski/ai-agent-doctor/blob/main/macierz_korelacji.txt

------------
👉 Uruchomienie aplikacji online:

https://ai-agent-doctor-klemanski.streamlit.app

------------
📂 Struktura plików:

ai-agent-doctor/

├── main.py                      # Główna logika aplikacji Streamlit

├── requirements.txt            # Biblioteki Pythona

├── model.pkl                   # Przetrenowany model uczenia maszynowego

├── scaler.pkl                  # Narzędzie do normalizacji danych modelu ML

├── train_model.py              # Skrypt do trenowania modelu

├── dane_rekomendacyjne_500.csv # Sztucznie wygenerowane dane wejściowe

├── static/images              # Folder z wykorzystanymi grafikami

├── corr.png                    # Grafika z macierzą korelacji

├── macierz_korelacji.txt       # Analiza i wnioski z macierzy korelacji

└── readme.md                   # Ten plik


------------
⚙️ Instalacja i uruchomienie aplikacji lokalnie:

1. Sklonuj repozytorium: https://github.com/KamilLemanski/ai-agent-doctor

2. Zainstaluj wymagane biblioteki: pip install -r requirements.txt

3. Skonfiguruj klucz API OpenAI: Utwórz plik .streamlit/secrets.toml i dodaj do niego swój klucz API

4. Uruchom aplikację: streamlit run main.py

------------
🔐 Zmienne środowiskowe:

Lokalnie: Ustaw zmienną środowiskową OPENAI_API_KEY w pliku .streamlit/secrets.toml

Streamlit: Użyj wpudowanego Streamlit Secrets w ustawieniach aplikacji i dodaj klucz API

------------
☁️ Deployment na platformie Streamlit Cloud:

1. Połącz swoje repozytorium GitHub ze Streamlit Cloud.
   
2. Upewnij się, że w repozytorium znajdują się następujące pliki: main.py, requirements.txt, model.pkl, scaler.pkl, train_model.py oraz folder z grafikami.
   
3. W ustawieniach aplikacji Streamlit Cloud, w sekcji "Secrets", dodaj swój klucz API OpenAI jako OPENAI_API_KEY.
   
4. Streamlit Cloud automatycznie zainstaluje zależności Pythona z requirements.txt.
  
5. Aplikacja zostanie uruchomiona pod wygenerowanym adresem URL.

------------
📌 Przykład użycia:

1. Wypełnij formularz w lewej kolumnie, podając swoje dane (wiek, płeć, wyniki badań, styl życia).

2. Kliknij przycisk "🔍 Wykonaj analizę ryzyka".

3. Poczekaj na przetworzenie danych. Na ekranie pojawi się tabela "📋 Analiza wyników badań" z procentowym ryzykiem i komentarzem AI.

4. Poniżej tabeli, jeśli zostaną spełnione odpowiednie warunki, zobaczysz sekcję "🩺 Dodatkowa diagnoza i rekomendacje".

5. W prawej kolumnie, w sekcji "💬 Medyczny Chatbot", możesz zadać dodatkowe pytania dotyczące Twoich wyników.

------------
📝 Licencja:

© 2025 Kamil Lemański. Projekt stworzony w celach edukacyjnych i demonstracyjnych.

------------
🙏 Credits:

OpenAI (GPT-4o), 
Streamlit Cloud, 
Scikit-learn.
