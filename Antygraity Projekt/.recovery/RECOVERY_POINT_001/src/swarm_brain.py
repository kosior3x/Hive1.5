
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
        self.vector_db = {} # {word: {'vector': np.array, 'count': int, 'last_seen': int}}

        # B1. Resonance Graph (Pairwise connections)
        self.resonance_graph = {} # {(w1, w2): {'weight': float, 'last_seen': int}}
        self.cycle_count = 0
        self.resonance_limit = 1000 # Soft limit for density control

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

        # PERSISTENCE FILE
        self.memory_file = "swarm_ltm_store.pkl"
        self.load_memory()

    def save_memory(self):
        """Save Memory and Vector DB to disk"""
        import pickle
        try:
            data = {
                'ltm': self.memory,
                'vector_db': self.vector_db,
                'resonance_graph': self.resonance_graph,
                'cycle_count': self.cycle_count
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 [Brain] Memory saved to {self.memory_file} (Vectors: {len(self.vector_db)}, Links: {len(self.resonance_graph)})")
        except Exception as e:
            print(f"❌ [Brain] Save failed: {e}")

    def load_memory(self):
        """Load Memory from disk"""
        import pickle
        import os
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory = data.get('ltm', self.memory)
                    self.vector_db = data.get('vector_db', self.vector_db)
                    self.resonance_graph = data.get('resonance_graph', {})
                    self.cycle_count = data.get('cycle_count', 0)
                print(f"📂 [Brain] Memory loaded. Known concepts: {len(self.vector_db)} | Resonance links: {len(self.resonance_graph)}")
            except Exception as e:
                print(f"⚠️ [Brain] Load failed, starting fresh: {e}")


    def update_memory(self, text, semantic_engine=None):
        """Enhanced with better Vector DB handling"""
        text_lower = text.lower()
        updates = []

        # ═══════════════════════════════════════════════════════
        # 1. VECTOR LEARNING (Hive Mind Learning)
        # ═══════════════════════════════════════════════════════
        if semantic_engine:
            # Extract words (3+ chars, Polish alphabet). Removing \b to ensure 'ę' at end is captured.
            # Fixed: Added 'ę' to the character set
            words = re.findall(r'[a-ząćęłńóśźż]{3,}', text_lower)

            # MINIMAL BLACKLIST (only true noise)
            blacklist_vec = {
                'jest', 'się', 'dla', 'lub', 'jak', 'czy', 'nie',
                'było', 'była', 'byli', 'tego', 'tych', 'tym', 'ten', 'tej',
            }

# ... (omitting unchanged middle parts for brevity, tool handles replacement by context if closely matched, but I will target specific blocks to be safe)
# Actually, I should do two replaces or one big one? The instruction says "Fix regex at line 119... and expand LTM". These are far apart.
# I will use multi_replace_file_content.

            blacklist_vec = {
                'jest', 'się', 'dla', 'lub', 'jak', 'czy', 'nie',
                'było', 'była', 'byli', 'tego', 'tych', 'tym', 'ten', 'tej',
            }

            # MUST-LEARN (always capture these)
            priority_words = {
                'python', 'pythonie', 'programowanie', 'kod', 'kodowanie',
                'rower', 'rowery', 'rowerze', 'roweru',
                'projekt', 'projektem', 'projekty', 'projektów',
                'algorytm', 'algorytmy', 'algorytmów',
                'chaos', 'chaosem', 'chaotyczny',
                'neuron', 'neurony', 'neuronów', 'neuronowy',
            }

            learned_this_msg = []

            for w in words:
                # Skip blacklist UNLESS it's priority
                if w in blacklist_vec and w not in priority_words:
                    continue

                # Encode
                vec = semantic_engine.encode(w)

                # Store or update
                if w not in self.vector_db:
                    # 4.2 Selection: Semantic Filter (A2/Selection)
                    # Only add if it relates to existing knowledge OR it's a fresh DB
                    # This prevents noise accumulation.
                    if len(self.vector_db) > 5:
                        import numpy as np
                        sims = [np.dot(vec, d['vector']) / (np.linalg.norm(vec) * np.linalg.norm(d['vector']) + 1e-9) for d in self.vector_db.values()]
                        max_sim = max(sims)
                        if max_sim < 0.3 and w not in priority_words:
                            # print(f"   ⚠️ [Brain] Ignored unrelated concept: '{w}' (Sim: {max_sim:.2f})")
                            continue

                    self.vector_db[w] = {'vector': vec, 'count': 1, 'last_seen': self.cycle_count}
                    learned_this_msg.append(w)
                    print(f"   🧠 [Vector DB] NEW: '{w}'")
                else:
                    self.vector_db[w]['count'] += 1
                    self.vector_db[w]['last_seen'] = self.cycle_count
                    print(f"   🧠 [Vector DB] +1: '{w}' (total: {self.vector_db[w]['count']})")

            # Status report
            if learned_this_msg:
                print(f"\n   📊 Nowe pojęcia: {len(learned_this_msg)} | Total: {len(self.vector_db)}")

        # ═══════════════════════════════════════════════════════
        # 2. FACT EXTRACTION (Multi-pass)
        # ═══════════════════════════════════════════════════════

        blacklist = ['jest', 'to', 'się', 'ja', 'ty', 'co', 'dzisiaj', 'robimy',
                     'sprawdzić', 'zrobić', 'będzie', 'móc', 'chcę', 'mamy',
                     'dziś', 'dobrze', 'źle', 'super', 'okej', 'samochód', 'auto']

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

        if updates or (semantic_engine and learned_this_msg):
            self.save_memory() # Auto-save on new info

        if updates:
            return "Zapamiętałem: " + ", ".join(updates)
        return None

    def prune_memory(self):
        """B1. Resonance Control & B2. Word Decay"""
        # 1. Decay resonance weights (B1)
        decay_rate = 0.05
        threshold = 0.15 # Lowered threshold slightly

        pruned_links = 0
        new_graph = {}

        for pair, data in self.resonance_graph.items():
            # Don't decay if recently active (B1.4) or very strong
            if self.cycle_count - data['last_seen'] < 20 or data['weight'] > 0.9:
                new_graph[pair] = data
            else:
                data['weight'] *= (1 - decay_rate)
                if data['weight'] >= threshold:
                    new_graph[pair] = data
                else:
                    pruned_links += 1

        self.resonance_graph = new_graph

        # 2. Word Decay (B2: Ochrona nisz - slower decay for rare/new words)
        words_to_prune = []
        for word, data in self.vector_db.items():
            # age_in_cycles handles the time since last engagement
            age_in_cycles = self.cycle_count - data.get('last_seen', 0)

            # Niche protection: if count is low (<5 hits), give it 2x more time before pruning
            # Rare concepts are protected longer to allow them to find a resonance niche
            is_niche = data.get('count', 1) < 5
            max_age = 1000 if is_niche else 500

            if age_in_cycles > max_age:
                words_to_prune.append(word)

        for w in words_to_prune:
            del self.vector_db[w]

        # 3. Check density soft limit (B1.6)
        if len(self.resonance_graph) > self.resonance_limit:
            print(f"   🧹 [Prune] Density limit reached. Applying aggressive pruning.")
            sorted_links = sorted(self.resonance_graph.items(), key=lambda x: x[1]['weight'], reverse=True)
            self.resonance_graph = dict(sorted_links[:int(self.resonance_limit * 0.8)])

        if pruned_links > 0 or words_to_prune:
            print(f"   🧹 [Prune] Removed {pruned_links} links, {len(words_to_prune)} dead words.")

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
        """Find concepts in LTM that resonate with current text, with Semantic Separation (A2) and Niche Protection (B2)"""
        if not semantic_engine or not self.vector_db: return ""

        import numpy as np
        current_words = re.findall(r'[a-ząćęłńóśźż]{3,}', text.lower())
        current_vec = semantic_engine.encode(text)

        best_word = ""
        best_sim = 0.0

        # B1. Cycle tracking
        self.cycle_count += 1

        for db_word, data in self.vector_db.items():
            if db_word in text.lower(): continue # Skip words present in input

            vec = data['vector']
            # Cosine similarity
            dot = np.dot(current_vec, vec)
            norm = np.linalg.norm(current_vec) * np.linalg.norm(vec) + 1e-9
            sim = dot / norm

            # Normalize -1..1 to 0..1
            sim = (sim + 1) / 2

            # B2. Niche Protection: Premium for rare/newly learned concepts
            usage_count = data.get('count', 1)
            # Inverse frequency bonus: Rare concepts get a boost to overcome dominant clusters
            niche_bonus = 1.0 + (1.0 / (usage_count + 1))
            sim *= niche_bonus

            # A2. Semantic Separation / Inhibition
            # "Inhibicja międzykategoriowa" - punish weak associations
            if sim < 0.6:
                sim *= 0.5
            else:
                # Temperature for close concepts (sharpen)
                sim = sim ** 2

            if sim > best_sim:
                best_sim = sim
                best_word = db_word

        # Limit global resonance gain
        if best_sim > 0.75:
            # B1. Track Resonance Graph
            for user_word in current_words:
                if user_word in self.vector_db:
                    # Sort pair to ensure (a,b) is same as (b,a)
                    pair = tuple(sorted((user_word, best_word)))
                    if pair not in self.resonance_graph:
                        self.resonance_graph[pair] = {'weight': 0.5, 'last_seen': self.cycle_count}
                    else:
                        self.resonance_graph[pair]['weight'] = min(1.0, self.resonance_graph[pair]['weight'] + 0.1)
                        self.resonance_graph[pair]['last_seen'] = self.cycle_count

            return f"\n🌌 Rezonans wektorowy: Twoja wypowiedź kojarzy mi się z pojęciem '{best_word}' (siła: {best_sim:.2f})"
        return ""

    def generate_response(self, category, chaos_energy, user_text, agents=None, semantic_engine=None):
        # Trigger pruning every 20 cycles
        if self.cycle_count % 20 == 0 and self.cycle_count > 0:
            self.prune_memory()

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
                total_concepts = len(self.vector_db)
                # Show top concepts by usage count (Expanded view)
                top_concepts = sorted(self.vector_db.items(), key=lambda x: x[1]['count'], reverse=True)[:40]
                concepts_str = ", ".join([f"{k}({v['count']})" for k,v in top_concepts])

                if total_concepts > 40:
                    concepts_str += f"\n... i {total_concepts - 40} innych."

                mem_parts.append(f"🧠 Baza Wektorowa [{total_concepts} pojęć]:\n{concepts_str}")

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
