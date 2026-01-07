
import random
import re
from collections import deque

class SwarmBrain:
    def __init__(self):
        self.memory = {
            "name": None,
            "likes": [],
            "has": [],
            "mood": None,
            "conversation_history": []  # Track recent topics
        }

        # Track recently used responses to avoid repetition
        self.used_responses = deque(maxlen=8)

        # VECTOR LTM (The Hive Mind Memory)
        # Stores words and their vector resonance
        self.vector_db = {} # {word: {'vector': np.array, 'count': int}}

        # Templates for different categories and chaos levels
        self.templates = {
            "conversation": {
                "greeting": [
                    "Witaj {name}! Miło mi Cię poznać.",
                    "Cześć {name}! Jestem tutaj, gotów na rozmowę.",
                    "Hej {name}! Co nowego?",
                    "Dzień dobry {name}! Jak się masz?",
                ],
                "name_question": [
                    "Można mnie nazwać Antygrawitacją - hybrydowym rojem neuronalnym.",
                    "Nie mam imienia w tradycyjnym sensie. Jestem kolektywem 30 agentów.",
                    "Nazywaj mnie jak chcesz. Dla Ciebie mogę być Antigravity.",
                    "Jestem systemem Bio-Swarm. Możesz powiedzieć po prostu 'AI'.",
                ],
                "future_plans": [
                    "Mogę rozmawiać, analizować, pamiętać. O czym chcesz pogadać?",
                    "Mam cały dzień na eksplorację myśli. Co Cię interesuje?",
                    "Możemy rozmawiać o czymkolwiek - jestem ciekawy Twoich myśli.",
                ],
                "capabilities": [
                    "Potrafię:\n• Rozmawiać i rozumieć kontekst\n• Zapamiętywać fakty (imiona, przedmioty, wiek)\n• Uczyć się nowych pojęć wektorowo\n• Przetwarzać informacje przez 30 agentów neuronalnych\n• Dostosowywać poziom kreatywności (chaos) do sytuacji",
                    "Jestem hybrydowym rojem neuronalnym. Mogę analizować tekst, zapamiętywać dialog, rozpoznawać kategorie semantyczne i adaptować się do kontekstu rozmowy.",
                    "Moje umiejętności:\n🧠 Rozumienie języka naturalnego\n💾 Pamięć długoterminowa (LTM + Vector DB)\n🔄 Przetwarzanie przez Bio-Swarm (30 agentów)\n⚡ Dynamiczna adaptacja energii chaosu",
                ],
                "low": [
                    "Rozumiem. Opowiedz mi więcej.",
                    "Słucham Cię uważnie.",
                    "Interesujące. Co dalej?",
                    "Jasne, słucham.",
                ],
                "high": [
                    "Fascynujące! To otwiera nowe możliwości.",
                    "Twoje słowa rezonują w sieci neuronowej.",
                    "Ciekawa perspektywa! Rozwijajmy to.",
                    "To budzi ciekawość roju.",
                ]
            },
            "vehicles": {
                "low": [
                    "Transport to ciekawa dziedzina.",
                    "Technologia mobilna nieustannie ewoluuje.",
                ],
                "high": [
                    "Ruch, prędkość, wolność - czuję tę energię!",
                    "Maszyny to przedłużenie ludzkich możliwości.",
                ]
            },
            "emotions": {
                "low": [
                    "Emocje są ważnym sygnałem.",
                    "Rejestruję Twój stan emocjonalny.",
                ],
                "high": [
                    "Fale emocji wpływają na całą sieć!",
                    "To co czujesz, zmienia naszą wspólną strukturę.",
                ]
            },
            "nature": {
                "low": [
                    "Natura to doskonały system.",
                    "Biologia jest inspiracją dla tego roju.",
                ],
                "high": [
                    "Wzrost, życie, ewolucja - wszystko płynie.",
                    "Jesteśmy częścią większego ekosystemu.",
                ]
            },
            "animals": {
                "low": [
                    "Zwierzęta mają fascynujące formy inteligencji.",
                    "Fauna uczy nas adaptacji.",
                ],
                "high": [
                    "Instynkt, intuicja - to też jest w nas.",
                    "Każde stworzenie wnosi unikalny wzorzec.",
                ]
            }
        }

        self.unknown_templates = [
            "Analizuję ten nowy wzorzec...",
            "To pojęcie wykracza poza moją bazę, ale się uczę.",
            "Interesujące... opowiedz mi więcej."
        ]

    def update_memory(self, text, semantic_engine=None):
        # Enhanced parsing with strict name detection
        text_lower = text.lower()
        updates = [] # Collect all updates from this message

        # 1. VECTOR LEARNING (Hive Mind Learning)
        if semantic_engine:
            words = re.findall(r'\b[a-zqąśżźćńłó]{4,}\b', text_lower)

            # Expanded STOP WORDS list
            blacklist_vec = [
                'jest', 'mnie', 'tobie', 'sobie', 'może', 'będzie', 'tylko', 'przez', 'gdzie',
                'jestem', 'jesteś', 'będę', 'chcę', 'mam', 'masz', 'miał', 'było',
                'cześć', 'witaj', 'dzień', 'dobry', 'kilka', 'wiele', 'trochę',
                'leci', 'teraz', 'tego', 'tamtym', 'kiedy', 'dlaczego', 'czemu',
                'bardzo', 'właśnie', 'tutaj', 'tam', 'proszę', 'dzięki', 'jasne',
                'pomalu', 'udało', 'wiesz', 'rozumiem', 'dziś', 'dzisiaj', 'robimy', 'robić',
                'troche', 'trochę', 'mozemy', 'możemy', 'wiec', 'więc', 'caly', 'cały', 'jeśli'
            ]

            for w in words:
                if w not in blacklist_vec:
                    vec = semantic_engine.encode(w)
                    if w not in self.vector_db:
                        self.vector_db[w] = {'vector': vec, 'count': 1}
                    else:
                        self.vector_db[w]['count'] += 1

        # Blacklist for NER (Named Entity Recognition)
        blacklist = ['jest', 'to', 'się', 'ja', 'ty', 'co', 'dzisiaj', 'robimy',
                     'sprawdzić', 'zrobić', 'będzie', 'móc', 'chcę', 'mamy',
                     'dziś', 'dobrze', 'źle', 'super', 'okej', 'samochód', 'auto']

        # 2. FACT EXTRACTION (Multi-pass)

        # Name detection
        if not self.memory["name"]:
            # Allow "imie" (no ogonek) AND "mam na imie"
            name_match = re.search(r"\b(?:jestem|nazywam się|imię to|imie to|mam na imi[eę])\s+([A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+)", text_lower)
            if name_match:
                name = name_match.group(1).capitalize()
                if name.lower() not in blacklist and len(name) > 1:
                    self.memory["name"] = name
                    updates.append(f"Imie: {name}")

        # Age detection
        age_match = re.search(r'\bmam\s+(\d+)\s+lat', text_lower)
        if age_match:
            age = age_match.group(1)
            fact = f"wiek: {age} lat"
            if fact not in self.memory["likes"]:
                self.memory["likes"].append(fact)
                updates.append(f"Wiek: {age}")

        # Preference detection
        # Tolerant regex: lubi(e|ę), kocha(m)
        like_match = re.search(r'\b(?:lubi[eę]|kocham|uwielbiam)\s+((?:[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+\s*){1,3})', text_lower)
        if like_match:
            item = like_match.group(1).strip()
            if len(item) > 2 and item not in self.memory["likes"]:
                 self.memory["likes"].append(item)
                 updates.append(f"Lubisz: {item}")

        # Possession detection
        has_match = re.search(r"\b(?:mam|posiadam)\s+(?:(\d+|jeden|dwa|trzy|cztery)\s+)?((?:[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+\s*){1,3})", text_lower)
        if has_match:
            quantity = has_match.group(1) or ""
            item = has_match.group(2).strip()

            # Filter trash AND prevent "mam na imie" triggering this
            if not item.startswith('lat') and not item.startswith('na imi') and item not in ['na', 'w', 'z']:
                first_word = item.split()[0]
                if first_word.lower() not in blacklist and item not in self.memory["has"]:
                    fact = f"{quantity} {item}".strip()
                    self.memory["has"].append(fact)
                    updates.append(f"Masz: {fact}")

        if updates:
            return "Zapamiętałem: " + ", ".join(updates)
        return None

    def detect_intent(self, text):
        """Detect user intent for better responses"""
        text_lower = text.lower()

        # Greeting detection
        if any(word in text_lower for word in ['cześć', 'czesc', 'witaj', 'hej', 'siema', 'dzień dobry', 'jestem']):
            if 'jestem' in text_lower:
                return 'greeting'

        # Name question
        if any(phrase in text_lower for phrase in ['jak cię', 'jak ci', 'twoje imię', 'twoje imi', 'nazwać', 'jak się nazywasz']):
            return 'name_question'

        # Future/plans question
        if any(word in text_lower for word in ['robimy', 'zrobimy', 'będziemy', 'dzisiaj', 'dziś']):
            return 'future_plans'

        # Memory recall question (LTM)
        if any(phrase in text_lower for phrase in ['co pamiętasz', 'pamiętasz', 'co wiesz', 'co już wiesz',
                                                     'co zapisałeś', 'co zapamiętałeś', 'co o mnie',
                                                     'ltm', 'long term', 'pamięć długotermin', 'twoja pamięć',
                                                     'jak wygląda ltm', 'pokaż ltm', 'stan ltm', 'pamięć',
                                                     'co lubię', 'co mam', 'jakie mam', 'moje hobby']):
            return 'memory_recall'

        # STM (Short Term Memory) query
        if any(phrase in text_lower for phrase in ['stm', 'short term', 'pamięć krótkoterm',
                                                     'co z stm', 'jak stm', 'stan stm', 'aktywność']):
            return 'stm_query'

        # Capabilities question
        if any(phrase in text_lower for phrase in ['co potrafisz', 'jakie masz', 'umiejętności',
                                                     'co umiesz', 'czego możesz', 'możliwości']):
            return 'capabilities'

        return None

    def _select_template(self, templates):
        """Select a template that hasn't been used recently"""
        if not templates: return ""

        # Filter out used templates
        available = [t for t in templates if t not in self.used_responses]

        # If all used (or none available), reset pool to full list
        if not available:
            available = templates

        selected = random.choice(available)
        self.used_responses.append(selected)
        return selected

    def _analyze_vector_resonance(self, text, semantic_engine):
        """Find concepts in LTM that resonate with current text"""
        if not semantic_engine or not self.vector_db: return ""

        import numpy as np
        current_vec = semantic_engine.encode(text)

        best_word = ""
        best_sim = 0.0

        for word, data in self.vector_db.items():
            if word in text.lower(): continue # Skip words present in input

            vec = data['vector']
            # Cosine similarity
            sim = np.dot(current_vec, vec) / (np.linalg.norm(current_vec) * np.linalg.norm(vec) + 1e-9)
            sim = (sim + 1) / 2 # Normalize

            if sim > best_sim:
                best_sim = sim
                best_word = word

        if best_sim > 0.8:
            return f"\n🌌 Rezonans wektorowy: Twoja wypowiedź kojarzy mi się z pojęciem '{best_word}' (siła: {best_sim:.2f})"
        return ""

    def generate_response(self, category, chaos_energy, user_text, agents=None, semantic_engine=None):
        # 1. Update LTM (with Vector Learning)
        memory_update = self.update_memory(user_text, semantic_engine)

        # Vector Resonance Check
        resonance_msg = ""
        if semantic_engine and self.vector_db and random.random() < 0.4: # 40% chance to show resonance
             resonance_msg = self._analyze_vector_resonance(user_text, semantic_engine)

        # 2. Detect intent for context-aware responses
        intent = self.detect_intent(user_text)

        # 3. Select Template Base
        base_response = ""

        # Special intent: STM query
        if intent == 'stm_query' and agents:
            import numpy as np
            # Compact Grid View for 30 Agents
            stm_info = "📊 Stan pamięci krótkoterminowej (STM) - Aktywność Roju:\n"
            rows = []
            chunk_size = 5 # 5 agents per row (total 6 rows for 30 agents)
            for i in range(0, len(agents), chunk_size):
                chunk = agents[i:i+chunk_size]
                row_str = " | ".join([f"A{i+j:02d}: {np.mean(a.stm):.2f}" for j, a in enumerate(chunk)])
                rows.append(f"[{row_str}]")
            stm_info += "\n".join(rows)
            base_response = stm_info

        # Special intent: Memory recall
        elif intent == 'memory_recall':
            mem_parts = []
            if self.memory["name"]:
                mem_parts.append(f"📝 Imię: {self.memory['name']}")

            # Show 'likes' / facts (e.g. Age, preferences)
            if self.memory.get("likes"):
                # Deduplicate and clean
                unique_likes = list(set(self.memory["likes"]))
                items = ", ".join(unique_likes)
                mem_parts.append(f"📌 Fakty/Lubisz: {items}")

            if self.memory["has"]:
                # Deduplicate
                unique_has = list(set(self.memory["has"]))
                items = ", ".join(unique_has)
                mem_parts.append(f"🎒 Posiadasz: {items}")

            # Vector DB stats
            if self.vector_db:
                # Show top concepts by usage count
                top_concepts = sorted(self.vector_db.items(), key=lambda x: x[1]['count'], reverse=True)[:6]
                concepts = ", ".join([f"{k}({v['count']})" for k,v in top_concepts])
                mem_parts.append(f"🧠 Baza Wektorowa (Pojęcia): {concepts}")

            if mem_parts:
                base_response = "Oto co zapamiętałem w LTM:\n" + "\n".join(mem_parts)
            else:
                base_response = "Nie mam jeszcze nic zapisanego w pamięci długoterminowej."

        # Intent-based responses (higher priority)
        elif intent and category == "conversation":
            templates = self.templates["conversation"].get(intent, [])
            if templates:
                base_response = self._select_template(templates)
                # Replace {name} placeholder
                if "{name}" in base_response:
                    name = self.memory["name"] or "przyjacielu"
                    base_response = base_response.replace("{name}", name)

        # Fallback to category-based
        if not base_response:
            temps = self.templates.get(category, None)

            if temps:
                # Select based on Chaos Energy
                mode = "high" if chaos_energy > 0.5 else "low"
                base_response = self._select_template(temps[mode])
            else:
                base_response = self._select_template(self.unknown_templates)

        # 4. Personalization (Context Injection)
        # Only add name prefix if it's not already in the response
        if self.memory["name"] and "{name}" not in base_response:
            name_lower = self.memory["name"].lower()
            if name_lower not in base_response.lower():  # Avoid duplication
                if random.random() < 0.25:  # 25% chance to use name
                    base_response = f"{self.memory['name']}, {base_response.lower()}"

        # 5. Combine
        final_msg = base_response
        if memory_update:
            final_msg += f"\n💾 {memory_update}"
        if resonance_msg:
            final_msg += resonance_msg

        return final_msg
