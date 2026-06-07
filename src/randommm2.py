# merged_bert_optuna_shap_fast.py
import re
import os
import time
import unicodedata
import numpy as np
import pandas as pd
import joblib
import optuna
import torch
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Settings (tweak for speed)
# ----------------------------
RND = 42
CONF_THRESHOLD = 0.65
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

# BERT config
BERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 64
BATCH_SIZE = 32

# SHAP config (smaller default to save time)
SHAP_GLOBAL_SAMPLE = 500

# Optuna config (keep quick by default)
OPTUNA_TRIALS = 3
OPTUNA_TIMEOUT = 600  # seconds (optional). Set None to disable.

# Embedding cache file
EMB_FILE = "artifacts/bert_embeddings.npy"

# ----------------------------
# Device
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# 1) Load data + preprocess
# ----------------------------
print("Loading data...")
df = pd.read_csv(r"data\newdata.csv")
df = df.dropna(subset=['text', 'label'])

def normalize_text(s):
    return unicodedata.normalize('NFKC', str(s)).strip()

df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# Label encoding required for XGBoost
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])
classes = label_encoder.classes_
num_classes = len(classes)
print("Classes:", classes)

# ----------------------------
# 2) BERT Embeddings with caching
# ----------------------------
os.makedirs("artifacts", exist_ok=True)

def compute_and_cache_embeddings(texts):
    print("Loading DistilBERT tokenizer & model...")
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = DistilBertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model.eval()

    all_emb = []
    n = len(texts)
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE].tolist()
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = bert_model(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls_emb)
        if i % (BATCH_SIZE*10) == 0:
            print(f"  BERT: processed {i}/{n}", end='\r')
    emb = np.vstack(all_emb)
    np.save(EMB_FILE, emb)
    print(f"\nSaved embeddings to {EMB_FILE}")
    return emb

# Try load cached embeddings first
if os.path.exists(EMB_FILE):
    try:
        print("Loading cached embeddings...")
        X_bert = np.load(EMB_FILE)
        if X_bert.shape[0] != len(df):
            print("Embedding count mismatch — recomputing embeddings.")
            X_bert = compute_and_cache_embeddings(df['text_lower'])
    except Exception as e:
        print("Failed to load embeddings (will recompute):", e)
        X_bert = compute_and_cache_embeddings(df['text_lower'])
else:
    X_bert = compute_and_cache_embeddings(df['text_lower'])

print("BERT shape:", X_bert.shape)

# ----------------------------
# 3) Faster regex features (cheap)
#    Use a fast regex phone flag for training.
#    Keep expensive phonenumbers checks only interactive.
# ----------------------------
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE = re.compile(r'\b\d{5}-\d{7}-\d\b')
# simple phone regex for dataset-level flag (fast)
PHONE_SIMPLE_RE = re.compile(r'(?:(?:\+?\d{1,3}[-\s]?)?\d{7,12})')

def has_email(text): return int(bool(EMAIL_RE.search(text or "")))
def has_cnic(text): return int(bool(CNIC_RE.search(text or "")))
def has_phone_simple(text): return int(bool(PHONE_SIMPLE_RE.search(text or "")))

def _near_context(text, i, j, win=16):
    lo = max(0, i - win); hi = min(len(text), j + win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(ndigits): return len(set(ndigits)) <= 2

# expensive strict phone check kept for interactive use only
def extract_valid_phones_strict(text):
    out = []
    try:
        for m in phonenumbers.PhoneNumberMatcher(text, DEFAULT_REGION):
            num = m.number
            if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)):
                continue
            if phonenumbers.region_code_for_number(num) not in ALLOWED_REGIONS:
                continue
            if phonenumbers.number_type(num) not in ALLOWED_TYPES:
                continue
            nd = str(num.national_number)
            if not (8 <= len(nd) <= 12) or len(nd) == 6 or _repetitive(nd):
                continue
            raw = m.raw_string.strip()
            if not raw.startswith('+') and not _near_context(text, m.start, m.end):
                continue
            out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    except Exception:
        pass
    return out

def has_phone_strict(text): return int(bool(extract_valid_phones_strict(text)))

# Apply cheap flags for training (fast)
df['email_flag'] = df['text_lower'].apply(has_email)
df['cnic_flag'] = df['text_lower'].apply(has_cnic)
df['phone_flag'] = df['text_lower'].apply(has_phone_simple)  # cheap

# ----------------------------
# 4) Combine features & split
# ----------------------------
X_regex = df[['email_flag', 'phone_flag', 'cnic_flag']].values   # (n,3)
X_final = np.hstack([X_bert, X_regex])                         # (n, 768+3)
y = df['label_num'].values

print("Final feature shape:", X_final.shape)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_final, y, test_size=0.3, stratify=y, random_state=RND
)

# ----------------------------
# 5) Optuna hyperparam search (fast config)
# ----------------------------
print("\n--- Starting Optuna ---")
def objective(trial):
    param = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': RND,
        'n_jobs': -1,
        'num_class': num_classes,
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    clf = xgb.XGBClassifier(**param)
    # use cv=3 but set n_jobs=1 to avoid nested parallelism slowdowns
    score = cross_val_score(clf, X_tr, y_tr, cv=3, scoring='accuracy', n_jobs=1).mean()
    return score

study = optuna.create_study(direction='maximize')
if OPTUNA_TIMEOUT:
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
else:
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
print("Best params:", study.best_params)

# ----------------------------
# 6) Train final XGBoost using best params
# ----------------------------
print("\nTraining final XGBoost...")
best_params = study.best_params.copy()
best_params.update({
    'objective': 'multi:softprob',
    'tree_method': 'hist',
    'random_state': RND,
    'n_jobs': -1,
    'num_class': num_classes
})
final_model = xgb.XGBClassifier(**best_params)
t0 = time.time()
final_model.fit(X_tr, y_tr)
print(f"Trained final model in {time.time()-t0:.1f}s")

# Save model + label encoder
os.makedirs("artifacts", exist_ok=True)
joblib.dump(final_model,   "artifacts/xgboost_bert_optuna_fast.joblib")
joblib.dump(label_encoder, "artifacts/label_encoder.joblib")
print("Saved artifacts/")

# ----------------------------
# 7) Evaluation
# ----------------------------
preds_num = final_model.predict(X_te)
preds_str = label_encoder.inverse_transform(preds_num)
y_te_str = label_encoder.inverse_transform(y_te)
print("\n=== EVAL ===")
print("Accuracy:", accuracy_score(y_te_str, preds_str))
print(classification_report(y_te_str, preds_str, digits=4))

# ----------------------------
# 8) SHAP: Global explanation (sampled)
# ----------------------------
print("\nComputing SHAP (sampled, may still take time)...")
explainer = None
try:
    explainer = shap.TreeExplainer(final_model)
    n_samples = min(len(X_te), SHAP_GLOBAL_SAMPLE)
    if len(X_te) > n_samples:
        sample_idx = np.random.choice(len(X_te), n_samples, replace=False)
        X_shap = X_te[sample_idx]
    else:
        X_shap = X_te

    shap_values = explainer.shap_values(X_shap)
    bert_dim = X_bert.shape[1]
    FEATURE_NAMES = [f"bert_{i}" for i in range(bert_dim)] + ['email_flag', 'phone_flag', 'cnic_flag']

    try:
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_NAMES, show=False)
        plt.tight_layout()
        plt.savefig("artifacts/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved artifacts/shap_summary.png")
    except Exception as e:
        print("Could not render full SHAP summary plot:", e)
except Exception as e:
    print("SHAP global explanation failed:", e)

# ----------------------------
# 9) Interactive loop with local SHAP explanation
# ----------------------------
def _canon(s): return re.sub(r'\s+', '', str(s or '').lower())
def _is_phone_label(name): return bool(re.search(r'phone', _canon(name)))
def _is_none_label(name): return _canon(name) in {'none','no_pii','nopii','neutral','other','negative'}

phone_idx = [i for i, c in enumerate(classes) if _is_phone_label(c)]

print("\n" + "="*60)
print("Interactive PII detection + SHAP explanations (fast mode)")
print("Type text, or 'exit' to quit.")
print("="*60)

# load tokenizer & bert model once for interactive step (if not loaded)
tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = DistilBertModel.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model.eval()

while True:
    s = input("Enter text: ")
    if s.lower().strip() in {"exit", "quit"}:
        break
    if not s.strip():
        continue

    # cheap regex flags
    email_flag = has_email(s)
    phone_flag = has_phone_simple(s)   # training flag
    cnic_flag = has_cnic(s)

    # Single BERT embed
    inputs = tokenizer([s.lower()], return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = bert_model(**inputs)
    emb = out.last_hidden_state[:, 0, :].cpu().numpy()   # shape (1,768)
    X_input = np.hstack([emb, np.array([[email_flag, phone_flag, cnic_flag]])])  # (1, 771)

    # Strict phone override (expensive) - do only in interactive mode
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE")
        print("Phones (E.164):", ", ".join(phones))
        print("-"*40)
        continue

    proba = final_model.predict_proba(X_input)[0]
    for i in phone_idx:
        proba[i] = 0.0
    top_i = int(np.argmax(proba))
    confidence = proba[top_i]
    pred_label = classes[top_i]
    final = pred_label if confidence >= CONF_THRESHOLD else 'none'

    if _is_none_label(final):
        print(f"No sensitive information detected. (confidence {confidence:.2f})")
    else:
        print(f"Detected Possible PII: {final.upper()} (confidence {confidence:.2f})")

        # Local SHAP explanation (fast top-K)
        if explainer is not None:
            try:
                shap_local = explainer.shap_values(X_input)
                local_vals = shap_local[top_i][0]
                K = 10
                topk_idx = np.argsort(-np.abs(local_vals))[:K]
                bert_dim = X_bert.shape[1]
                print("\nTop contributing features:")
                for idx in topk_idx:
                    fname = FEATURE_NAMES[idx]
                    val = local_vals[idx]
                    kind = "BERT-dim" if idx < bert_dim else "Regex-flag"
                    print(f"  {fname:>20s} | impact {val:+.4f} | {kind}")
            except Exception as e:
                print("Could not compute local SHAP explanation:", e)
    print("-"*40)

print("Exiting. Goodbye!")
