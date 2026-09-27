import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
from pathlib import Path

# =========================
# CHEMINS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_best.joblib"

# =========================
# CHARGEMENT DU MODELE
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# FENETRE PRINCIPALE
# =========================
root = tk.Tk()
root.title("Prédiction Hypertension")
root.geometry("1100x650")
root.configure(bg="#0f172a")
root.minsize(1000, 600)

# =========================
# STYLES SOMBRES
# =========================
style = ttk.Style()
style.theme_use("clam")

BG = "#0f172a"
PANEL = "#111827"
PANEL_2 = "#1f2937"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
ACCENT = "#38bdf8"
ACCENT_DARK = "#0284c7"
SUCCESS = "#10b981"
DANGER = "#ef4444"
WARNING = "#f59e0b"
WHITE = "#ffffff"

style.configure("TFrame", background=BG)
style.configure("Panel.TFrame", background=PANEL)
style.configure("Card.TFrame", background=PANEL_2)

style.configure(
    "TLabel",
    background=BG,
    foreground=TEXT,
    font=("Segoe UI", 10)
)

style.configure(
    "Title.TLabel",
    background=BG,
    foreground=WHITE,
    font=("Segoe UI", 20, "bold")
)

style.configure(
    "Subtitle.TLabel",
    background=BG,
    foreground=MUTED,
    font=("Segoe UI", 10)
)

style.configure(
    "PanelTitle.TLabel",
    background=PANEL,
    foreground=WHITE,
    font=("Segoe UI", 13, "bold")
)

style.configure(
    "Field.TLabel",
    background=PANEL,
    foreground=TEXT,
    font=("Segoe UI", 10)
)

style.configure(
    "TEntry",
    fieldbackground="#1e293b",
    foreground=TEXT,
    insertcolor=TEXT,
    bordercolor="#334155",
    lightcolor="#334155",
    darkcolor="#334155"
)

style.configure(
    "TCombobox",
    fieldbackground="#1e293b",
    background="#1e293b",
    foreground=TEXT,
    arrowcolor=WHITE
)

style.map(
    "TCombobox",
    fieldbackground=[("readonly", "#1e293b")],
    foreground=[("readonly", TEXT)]
)

style.configure(
    "Accent.TButton",
    background=ACCENT,
    foreground=WHITE,
    font=("Segoe UI", 11, "bold"),
    padding=10,
    borderwidth=0
)
style.map(
    "Accent.TButton",
    background=[("active", ACCENT_DARK)]
)

style.configure(
    "Secondary.TButton",
    background="#334155",
    foreground=WHITE,
    font=("Segoe UI", 10, "bold"),
    padding=8,
    borderwidth=0
)
style.map(
    "Secondary.TButton",
    background=[("active", "#475569")]
)

# =========================
# LAYOUT GLOBAL
# =========================
main = ttk.Frame(root)
main.pack(fill="both", expand=True, padx=20, pady=20)

header = ttk.Frame(main)
header.pack(fill="x", pady=(0, 15))

ttk.Label(header, text="Prédiction Hypertension", style="Title.TLabel").pack(anchor="w")
ttk.Label(
    header,
    text="Interface sombre de prédiction du risque avec SVM",
    style="Subtitle.TLabel"
).pack(anchor="w", pady=(4, 0))

content = ttk.Frame(main)
content.pack(fill="both", expand=True)

left_panel = ttk.Frame(content, style="Panel.TFrame", padding=20)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

right_panel = ttk.Frame(content, style="Panel.TFrame", padding=20)
right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

# =========================
# TITRES DE PANNEAUX
# =========================
ttk.Label(left_panel, text="Données du patient", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 15))
ttk.Label(right_panel, text="Résultat de prédiction", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 15))

# =========================
# FORMULAIRE
# =========================
form = ttk.Frame(left_panel, style="Panel.TFrame")
form.pack(fill="both", expand=True)

entries = {}

def add_entry(row, label, default=""):
    ttk.Label(form, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=8, padx=(0, 15))
    entry = ttk.Entry(form, width=30)
    entry.grid(row=row, column=1, sticky="ew", pady=8)
    if default:
        entry.insert(0, default)
    entries[label] = entry

def add_combo(row, label, values, default_index=0):
    ttk.Label(form, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=8, padx=(0, 15))
    combo = ttk.Combobox(form, values=values, state="readonly", width=28)
    combo.grid(row=row, column=1, sticky="ew", pady=8)
    combo.current(default_index)
    entries[label] = combo

form.columnconfigure(1, weight=1)

add_entry(0, "Age")
add_entry(1, "Salt_Intake")
add_entry(2, "Stress_Score")
add_entry(3, "Sleep_Duration")
add_entry(4, "BMI")

add_combo(5, "BP_History", ["Normal", "High"], 0)
add_combo(6, "Medication", ["None", "Beta Blocker","Diuretic","N²one","ACE Inhibitor"], 0)
add_combo(7, "Family_History", ["No", "Yes"], 0)
add_combo(8, "Exercise_Level", ["Low", "Moderate", "High"], 1)
add_combo(9, "Smoking_Status", ["No", "Yes"], 0)

# =========================
# BOUTONS
# =========================
button_bar = ttk.Frame(left_panel, style="Panel.TFrame")
button_bar.pack(fill="x", pady=(20, 0))

def clear_fields():
    for key, widget in entries.items():
        if isinstance(widget, ttk.Entry):
            widget.delete(0, tk.END)
        else:
            widget.current(0)

def set_result(title, prob, detail):
    result_title.config(text=title, foreground=SUCCESS if prob < 0.5 else DANGER)
    probability_value.config(text=f"{prob:.2%}")
    result_detail.config(text=detail)

def predict():
    try:
        data = {
            "Age": float(entries["Age"].get()),
            "Salt_Intake": float(entries["Salt_Intake"].get()),
            "Stress_Score": float(entries["Stress_Score"].get()),
            "Sleep_Duration": float(entries["Sleep_Duration"].get()),
            "BMI": float(entries["BMI"].get()),
            "BP_History": entries["BP_History"].get(),
            "Medication": entries["Medication"].get(),
            "Family_History": entries["Family_History"].get(),
            "Exercise_Level": entries["Exercise_Level"].get(),
            "Smoking_Status": entries["Smoking_Status"].get()
        }

        df = pd.DataFrame([data])
        proba = float(model.predict_proba(df)[0][1])
        pred = model.predict(df)[0]

        if isinstance(pred, str):
            pred = 1 if pred == "Yes" else 0
        else:
            pred = int(pred)

        if pred == 1:
            title = "Risque élevé d'hypertension"
            detail = "Le modèle estime que ce patient présente un risque important."
        else:
            title = "Risque faible d'hypertension"
            detail = "Le modèle estime que ce patient présente un risque modéré à faible."

        set_result(title, proba, detail)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

predict_btn = ttk.Button(button_bar, text="Prédire", style="Accent.TButton", command=predict)
predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

clear_btn = ttk.Button(button_bar, text="Effacer", style="Secondary.TButton", command=clear_fields)
clear_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

# =========================
# PANNEAU DROIT RESULTAT
# =========================
result_card = ttk.Frame(right_panel, style="Card.TFrame", padding=20)
result_card.pack(fill="both", expand=True)

result_title = tk.Label(
    result_card,
    text="En attente de prédiction",
    bg=PANEL_2,
    fg=WHITE,
    font=("Segoe UI", 16, "bold")
)
result_title.pack(anchor="center", pady=(15, 20))

tk.Label(
    result_card,
    text="Probabilité",
    bg=PANEL_2,
    fg=MUTED,
    font=("Segoe UI", 11)
).pack(anchor="center")

probability_value = tk.Label(
    result_card,
    text="--",
    bg=PANEL_2,
    fg=ACCENT,
    font=("Segoe UI", 34, "bold")
)
probability_value.pack(anchor="center", pady=(8, 20))

result_detail = tk.Label(
    result_card,
    text="Remplis les champs puis clique sur Prédire.",
    bg=PANEL_2,
    fg=TEXT,
    font=("Segoe UI", 11),
    wraplength=350,
    justify="center"
)
result_detail.pack(anchor="center", pady=(0, 20))

# =========================
# PETIT BLOC INFO
# =========================
info_box = ttk.Frame(result_card, style="Panel.TFrame", padding=15)
info_box.pack(fill="x", pady=(20, 0))

ttk.Label(
    info_box,
    text="Conseil",
    style="PanelTitle.TLabel"
).pack(anchor="w")

ttk.Label(
    info_box,
    text="Utilise des valeurs réalistes pour obtenir une prédiction cohérente.",
    style="Field.TLabel",
    wraplength=340
).pack(anchor="w", pady=(6, 0))

# =========================
# LANCEMENT
# =========================
root.mainloop()
# Application de prédiction de l'hypertension