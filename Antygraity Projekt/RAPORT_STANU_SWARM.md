# 📄 Raport Stanu Antygrawitacja AI (v3.3)

**Data i Czas:** 16.12.2025, 12:55
**Status:** Stabilny / W fazie Ewolucji
**Architektura:** Hybrid Bio-Swarm (30 Agentów)

---

## 1. 🏗️ Architektura Systemu

### 🐝 Rój (Bio-Swarm)
*   **Liczba Agentów:** 30 niezależnych jednostek.
*   **Struktura Agenta:**
    *   `STM` (Pamięć Krótkotrwała): Wektor 16D, reagujący na bodźce w czasie rzeczywistym.
    *   `LTM` (Pamięć Długotrwała): Baza faktów + Baza wektorowa.
    *   `Genome`: Unikalne cechy wpływające na poziom chaosu (`sigma`).
    *   `Decay`: Mechanizm zapominania (powrót do stanu bazowego `origin` z czasem).
*   **Wizualizacja:** Podgląd aktywności STM dla każdego agenta (siatka 6x5).

### 🧠 Mózg (SwarmBrain)
*   **Silnik Semantyczny:** Własny moduł wektorowy (FastText-lite logic), niezależny od spaCy/NLTK.
*   **Detekcja Intencji:** Regex-based + Context-aware (rozpoznaje powitania, pytania o pamięć, plany).
*   **Generowanie Odpowiedzi:** Hybrydowe (Szablony + Kontekst + Wstrzykiwanie Faktów).

---

## 2. 💾 Pamięć (Memory Systems)

### A. LTM Faktyczna (Rule-Based)
System potrafi ekstrahować i kategoryzować fakty w czasie rzeczywistym (również z zdań złożonych):
*   **Imię:** *"Jestem Kamil"*
*   **Wiek:** *"Mam 35 lat"*
*   **Zainteresowania:** *"Lubię lody"*
*   **Posiadanie:** *"Mam dwa koty"* (Naprawiono błąd, gdzie "Mam na imię" było traktowane jako posiadanie).

### B. LTM Wektorowa (Hive Mind Learning)
*   **Mechanizm:** Każde ważne słowo (>4 litery, nie będące "szumem") jest kodowane i zapisywane.
*   **Pojęcia:** System buduje własną mapę skojarzeń. Np. jeśli często mówisz o kodowaniu, wektor `python` będzie silnie powiązany z kontekstem rozmowy.
*   **Rezonans:** System potrafi powiedzieć: *"Twoja wypowiedź kojarzy mi się z pojęciem X"*.

### C. STM (Short-Term Memory)
*   Utrzymuje kontekst rozmowy przez ~20 cykli.
*   Każdy agent ma nieco inny "pogląd" na sytuację dzięki unikalnemu genomowi i losowym mutacjom chaosu.

---

## 3. ⚡ Energia Chaosu (Chaos Energy)
*   **Rola:** Decyduje o kreatywności odpowiedzi.
*   **Low Chaos (<0.5):** Odpowiedzi logiczne, krótkie, faktograficzne.
*   **High Chaos (>0.5):** Odpowiedzi abstrakcyjne, metaforyczne, nawiązujące do natury i ewolucji.
*   **Adaptacja:** Rój automatycznie dostosowuje poziom chaosu do tempa rozmowy.

---

## 4. 🛠️ Ostatnie Zmiany & Poprawki (Changelog)

1.  **Szlif NER (Imię):** Naprawiono błąd parsowania "mam na imię" jako przedmiotu w plecaku. Teraz poprawnie trafia do slotu `Imie`.
2.  **Multitasking LTM:** System przetwarza wiele faktów z jednego zdania (np. Imię + Wiek + Hobby naraz).
3.  **Wektorowy Odśmiecacz:** Rozszerzono `blacklist` o słowa-wypełniacze (*trochę, więc, możemy*), aby baza wiedzy była konkretna.
4.  **Optymalizacja Startu:** Usunięto zbędne sprawdzanie spaCy/NLTK, co przyspieszyło uruchamianie aplikacji.

---

## 5. 🚀 Roadmap (Co dalej?)

*   [ ] **Zapis do pliku:** Pełna persystencja LTM (zapis `vector_db` i `memory` do JSON po zamknięciu).
*   [ ] **Wizualizacja Wektorów:** Graficzne przedstawienie (np. matplotlib) jak blisko są pojęcia w "głowie" roju.
*   [ ] **Głębsza Ewolucja:** Agenci, którzy najlepiej dopasowują się do rozmówcy, powinni przekazywać swój genom dalej (algorytm genetyczny).

---
*Raport wygenerowany automatycznie przez Antygrawitacja AI.*
