import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import sys
import os
import json

# Add src and utils path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))
sys.path.append(parent_dir)

try:
    from swarm_hybrid_evolution import HybridSwarmController
    from swarm_acn import acn_tick
    from semantic_mock import SemanticEngineMock
    from swarm_brain import SwarmBrain
except ImportError as e:
    print(f"❌ Backend Import Error: {e}")
    # Fallback/Mock if running standalone without correct paths or missing deps
    print("⚠️ WARNING: Running in Fallback Mode (Backend not found).")
    class DummyAgent:
        def __init__(self):
            self.stm = [0.0]*16
            self.origin = [0.0]*16
        def process_cycle(self): pass

    class HybridSwarmController:
        def __init__(self, size=30):
            self.agents = [DummyAgent() for _ in range(size)]
            self.codec = None
        def run_experiment(self, text): pass

    class SemanticEngineMock:
        def encode(self, w): return [0.1]*16
        def decode_category(self, v): return "mock_cat", 0.99

    def acn_tick(agents): return None

    class SwarmBrain:
        def generate_response(self, cat, chaos, txt): return "System offline."

class AntigravityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity AI (Hybrid Swarm)")
        self.root.geometry("1000x700")
        self.root.configure(bg="#ffffff")

        # Fonts & Colors
        self.font_main = ("Segoe UI", 11)
        self.font_bold = ("Segoe UI", 11, "bold")
        self.font_small = ("Segoe UI", 9)
        self.col_bg = "#ffffff"
        self.col_sidebar = "#f7f7f8"
        self.col_accent = "#10a37f"

        # History Data
        self.history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.json")
        self.chats = self.load_history() # List of dicts: {'title': str, 'messages': []}
        self.current_chat_idx = None # Index of active chat

        # Initialize Swarm Backend
        self.init_swarm()

        # Initialize Brain for intelligent responses
        self.brain = SwarmBrain()



        # --- LAYOUT ---
        self.sidebar = tk.Frame(root, width=260, bg=self.col_sidebar)
        self.sidebar.pack(fill=tk.Y, side=tk.LEFT)
        self.sidebar.pack_propagate(False)

        # Sidebar Header (New Chat)
        lbl_title = tk.Label(self.sidebar, text="+ Nowy czat", font=("Segoe UI", 10, "bold"),
                             bg=self.col_sidebar, fg="#444", anchor="w", padx=15, pady=20, cursor="hand2")
        lbl_title.pack(fill=tk.X)
        lbl_title.bind("<Button-1>", lambda e: self.create_new_chat())

        # Session List
        self.session_list = tk.Listbox(self.sidebar, bg=self.col_sidebar, fg="#333", bd=0,
                                       font=self.font_main, highlightthickness=0, selectbackground="#e0e0e0", activestyle="none")
        self.session_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.session_list.bind('<<ListboxSelect>>', self.on_select_chat)

        # Update Sidebar Content
        self.refresh_sidebar()

        # User Profile
        user_frame = tk.Frame(self.sidebar, bg=self.col_sidebar, height=60)
        user_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=15)
        tk.Label(user_frame, text="Kamil Kos", font=self.font_bold, bg=self.col_sidebar).pack(side=tk.LEFT)
        tk.Label(user_frame, text="PRO", font=("Segoe UI", 8, "bold"), bg="#fae69e", fg="#555", padx=5).pack(side=tk.RIGHT)

        # Main Area
        self.main_area = tk.Frame(root, bg=self.col_bg)
        self.main_area.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)

        self.chat_display = scrolledtext.ScrolledText(self.main_area, font=self.font_main, bg=self.col_bg,
                                                      fg="#333", bd=0, padx=20, pady=20, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        self.chat_display.tag_config("user", justify="right", foreground="#333", rmargin=10)
        self.chat_display.tag_config("ai", foreground="#111")
        self.chat_display.tag_config("meta", foreground="#888", font=self.font_small)
        self.chat_display.tag_config("bold", font=self.font_bold)

        # Input Area
        input_frame = tk.Frame(self.main_area, bg=self.col_bg, pady=20)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.input_field = tk.Entry(input_frame, font=("Segoe UI", 12), bd=1, relief=tk.SOLID)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(40, 10), ipady=8)
        self.input_field.bind("<Return>", self.send_message)

        btn_send = tk.Button(input_frame, text=" ➤ ", font=("Arial", 12), bg=self.col_accent, fg="white",
                             bd=0, cursor="hand2", command=self.send_message)
        btn_send.pack(side=tk.RIGHT, padx=(0, 40), ipady=4)

        # Load initial chat
        if not self.chats:
            self.create_new_chat()
        else:
            self.load_chat(0) # Load most recent/top

    def init_swarm(self):
        try:
            self.swarm = HybridSwarmController(size=30)
            # Try to load FastText if available, else Mock
            try:
                from semantic_fasttext import FastTextEngine
                self.swarm.codec = FastTextEngine()
            except:
                self.swarm.codec = SemanticEngineMock()
            self.mock_sem_engine = self.swarm.codec
        except Exception as e:
            print(f"Backend Init Error: {e}")

    # --- HISTORY MANAGEMENT ---
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Save error: {e}")

    def refresh_sidebar(self):
        self.session_list.delete(0, tk.END)
        for chat in self.chats:
            title = chat.get('title', 'Bez tytułu')
            # Truncate if too long
            if len(title) > 25: title = title[:22] + "..."
            self.session_list.insert(tk.END, f"💬 {title}")

    def create_new_chat(self):
        self.chats.insert(0, {'title': 'Nowy czat', 'messages': []}) # Insert at top
        self.refresh_sidebar()
        self.load_chat(0)

    def on_select_chat(self, event):
        selection = self.session_list.curselection()
        if selection:
            idx = selection[0]
            self.load_chat(idx)

    def load_chat(self, index):
        self.current_chat_idx = index
        self.session_list.selection_clear(0, tk.END)
        self.session_list.selection_set(index)

        chat = self.chats[index]
        messages = chat.get('messages', [])

        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)

        if not messages:
            self.append_ai_message_ui("Witaj! Jestem gotowy do pracy.\nWpisz wiadomość, aby rozpocząć przetwarzanie przez rój.")
        else:
            for msg in messages:
                if msg['role'] == 'user':
                    self.append_user_message_ui(msg['text'])
                else:
                    self.append_ai_message_ui(msg['text'])

        self.chat_display.configure(state=tk.DISABLED)

    # --- MESSAGING ---
    def send_message(self, event=None):
        msg = self.input_field.get().strip()
        if not msg: return

        self.input_field.delete(0, tk.END)

        # UI Update
        self.append_user_message_ui(msg)

        # Save to history
        if self.current_chat_idx is not None:
            self.chats[self.current_chat_idx]['messages'].append({'role': 'user', 'text': msg})

            # Auto-title if it's the first message
            if len(self.chats[self.current_chat_idx]['messages']) == 1:
                new_title = msg[:30] # First 30 chars
                self.chats[self.current_chat_idx]['title'] = new_title
                self.refresh_sidebar()
                self.session_list.selection_set(self.current_chat_idx)

            self.save_history()

        # Process in thread
        threading.Thread(target=self.process_swarm, args=(msg,)).start()

    def process_swarm(self, text):
        # 1. Inject to ALL agents (prevent zero-pull)
        vec = self.mock_sem_engine.encode(text)
        for a in self.swarm.agents:
            a.stm = vec.copy()
            a.origin = vec.copy()

        # 2. Simulate Thinking (with ACN)
        start_time = time.time()
        total_energy = 0.0

        for i in range(10):
            # A. Processing
            for a in self.swarm.agents: a.process_cycle()

            # B. ACN Communication
            res = acn_tick(self.swarm.agents)
            if res:
                total_energy += res[2] if len(res) > 2 else 0

            time.sleep(0.05)

        elapsed = (time.time() - start_time) * 1000

        # 3. Decode
        out_vec = self.swarm.agents[-1].stm
        try:
            cat, sim = self.mock_sem_engine.decode_category(out_vec)
        except:
             # Fallback if method differs in Mock/FastText
             cat, sim = "unknown", 0.0

        # 4. Generate intelligent response using SwarmBrain
        # Pass semantic engine to enable Vector LTM
        brain_response = self.brain.generate_response(cat, total_energy, text, self.swarm.agents, self.mock_sem_engine)

        # Format with telemetry
        response = f"{brain_response}\n\n⚙️ Kontekst: {str(cat).upper()} | Dopasowanie: {sim:.2f} | Chaos: {total_energy:.3f}"

        # Save & UI Update
        if self.current_chat_idx is not None:
             self.chats[self.current_chat_idx]['messages'].append({'role': 'assistant', 'text': response})
             self.save_history()

        self.root.after(0, lambda: self.append_ai_message_ui(response))

    # --- UI HELPERS (No logic, just display) ---
    def append_user_message_ui(self, text):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.insert(tk.END, f"TY:\n{text}", "user")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def append_ai_message_ui(self, text):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.insert(tk.END, "Antygrawitacja AI ", "bold")
        self.chat_display.insert(tk.END, f"[{time.strftime('%H:%M')}]\n", "meta")
        self.chat_display.insert(tk.END, text, "ai")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)




    def on_closing(self):
        if hasattr(self, 'brain'):
            self.brain.save_memory()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AntigravityGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
