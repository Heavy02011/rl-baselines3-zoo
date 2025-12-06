# extract_best_grpo_mcc_v2.py
# model: GPT-5.1 Thinking, prompt: "wie nutze ich rl_zoo3.optimize / unable to open database file"

import optuna
from pathlib import Path

# Repo-Root = eine Ebene über "scripts"
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "grpo_mcc.db"

print("DB path:", DB_PATH)
print("Exists:", DB_PATH.exists())

storage = f"sqlite:///{DB_PATH}"

study = optuna.load_study(
    study_name="grpo_mcc",        # muss zu deinem --study-name passen
    storage=storage,
)

best = study.best_trial
print("Best trial number:", study.best_trial.number)

print("Best value:", best.value)
print("Best params:")
for k, v in best.params.items():
    print(f"{k}: {v}")

