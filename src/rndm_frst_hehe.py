import re
import pandas as pd
import numpy as np
import unicodedata
import os
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

# ----------------------------
# Settings
# ----------------------------

RND = 42
CONF_THRESHOLD = 0.65
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

# ----------------------------
# Load dataset
# ----------------------------

print("Loading Data...")
# Ensure this path is correct
df = pd.read_csv(r"data\newdata.csv")
df = df.dropna(subset=['text', 'label'])

def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# ----------------------------
# Label Encoding
# ----------------------------
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])

X_text = df['text_lower']
y = df['label_num']

X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=RND
)

# ----------------------------
# Regex-based feature functions
# ----------------------------

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE = re.compile(r'\b\d{5}-\d{7}-\d\b')

def has_email(text):
    return int(bool(EMAIL_RE.search(text or "")))

def has_cnic(text):
    return int(bool(CNIC_RE.search(text or "")))

def _near_context(text, i, j, win=16):
    lo = max(0, i - win)
    hi = min(len(text), j + win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(ndigits):
    return len(set(ndigits)) <= 2

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

def has_phone_strict(text):
    return int(bool(extract_valid_phones_strict(text)))

# Apply regex
df['email_flag'] = df['text_lower'].apply(has_email)
df['cnic_flag'] = df['text_lower'].apply(has_cnic)
df['phone_flag'] = df['text_lower'].apply(has_phone_strict)

# ----------------------------
# TF-IDF + feature stack
# ----------------------------

print("Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000)
X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

regex_tr = csr_matrix(df.loc[X_tr_text.index, ['email_flag', 'phone_flag', 'cnic_flag']].values)
regex_te = csr_matrix(df.loc[X_te_text.index, ['email_flag', 'phone_flag', 'cnic_flag']].values)

X_tr = hstack([X_tr_vec, regex_tr])
X_te = hstack([X_te_vec, regex_te])

# ----------------------------
# PREPARE FEATURE NAMES FOR SHAP
# ----------------------------
tfidf_feature_names = vectorizer.get_feature_names_out()
regex_feature_names = ['email_flag', 'phone_flag', 'cnic_flag']
all_feature_names = list(tfidf_feature_names) + regex_feature_names

print(f"Total features: {len(all_feature_names)}")

# ----------------------------
# Train XGBoost
# ----------------------------

model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    random_state=RND,
    n_jobs=-1,
    tree_method="hist"
)

print("Training XGBoost...")
model.fit(X_tr, y_tr)

# Eval
print("Predicting on test set...")
preds_num = model.predict(X_te)
preds_str = label_encoder.inverse_transform(preds_num)
y_te_str = label_encoder.inverse_transform(y_te)

print("\n=== XGBOOST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te_str, preds_str))
print("\nClassification Report:\n", classification_report(y_te_str, preds_str, digits=4))

# ----------------------------
# SHAP SETUP
# ----------------------------
print("\nInitializing SHAP Explainer (this may take a few seconds)...")
explainer = shap.TreeExplainer(model)

# ----------------------------
# Interactive PII Detection
# ----------------------------

def _canon(s):
    return re.sub(r'\s+', '', str(s or '').lower())

def _is_phone_label(name):
    return bool(re.search(r'phone', _canon(name)))

def _is_none_label(name):
    return _canon(name) in {'none', 'no_pii', 'nopii', 'neutral', 'other', 'negative'}

classes = label_encoder.classes_
phone_idx = [i for i, c in enumerate(classes) if _is_phone_label(c)]

ADDRESS_KEYWORDS = {
    'st', 'street', 'rd', 'road', 'lane', 'ave', 'avenue', 'blvd',
    'sector', 'block', 'phase', 'house', 'flat', 'apt', 'apartment',
    'town', 'colony', 'city', 'zip', 'postal', 'area', 'near', 'floor'
}
def is_plausible_address(text):
    text = text.lower().strip()
    if len(text) < 6 or text in ['hello', 'testing', 'nothing', 'random']: return False
    has_digit = any(char.isdigit() for char in text)
    tokens = set(re.split(r'\W+', text))
    has_keyword = not tokens.isdisjoint(ADDRESS_KEYWORDS)
    return True if (has_keyword or has_digit) else False

print("\n" + "="*60)
print("🤖 SHAP-ENHANCED INTERACTIVE PII DETECTION")
print("Type 'exit' to quit.")
print("="*60)

while True:
    s = input("\nEnter text to check: ")
    if s.lower().strip() in ["exit", "quit"]:
        break

    if not s.strip():
        continue

    # 1. Regex Checks
    email_flag = has_email(s)
    phone_flag = has_phone_strict(s)
    cnic_flag = has_cnic(s)
    
    # 2. Strict Phone 
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE")
        print("Phones:", ", ".join(phones))
        print("-" * 60)
        continue

    # 3. Prepare Input Vector
    X_text_input = vectorizer.transform([s])
    X_input = hstack([X_text_input, csr_matrix([[email_flag, phone_flag, cnic_flag]])])

    # 4. Predict
    proba = model.predict_proba(X_input)[0]
    
    # Suppress phone class if regex didn't find one
    for i in phone_idx:
        proba[i] = 0.0 

    top_i = int(np.argmax(proba))
    confidence = proba[top_i]
    pred_label = classes[top_i]

    # Address Heuristic Check
    if 'address' in _canon(pred_label) and not is_plausible_address(s):
        pred_label = 'none'
        confidence = 1.0

    # Apply Threshold
    final = pred_label if confidence >= CONF_THRESHOLD else 'none'

    # 5. Output Prediction
    if _is_none_label(final):
        print("Result: No sensitive information detected.")
    else:
        print(f"Result: Detected Possible PII: {str(final).upper()}")
        print(f"Confidence: {confidence:.2f}")

        # ----------------------------
        # SHAP EXPLANATION (SAFE VERSION)
        # ----------------------------
        # ----------------------------
        # SHAP EXPLANATION (FIXED & ROBUST)
        # ----------------------------
        print("\n🔍 WHY did the model predict this?")
        
        try:
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_input)
            
            # --- ROBUST SHAPE HANDLING ---
            class_shap_values = None
            
            # 1. Handle List (Multiclass) vs Array (Binary)
            if isinstance(shap_values, list):
                # If valid index, take that class, else take the first one
                idx_to_take = top_i if top_i < len(shap_values) else 0
                class_shap_values = shap_values[idx_to_take]
            else:
                class_shap_values = shap_values

            # 2. Handle Sparse Matrix (CSR)
            if hasattr(class_shap_values, "toarray"):
                class_shap_values = class_shap_values.toarray()
                
            # 3. CRITICAL FIX: Force Flatten to 1D Array
            # This ensures we have a flat list of numbers, not a 2D matrix
            class_shap_values = np.array(class_shap_values).reshape(-1)
            
            # ---------------------------------------------

            # Get indices of top 5 contributors (magnitude)
            # We sort by absolute value to see what mattered most (pos or neg)
            top_indices = np.argsort(np.abs(class_shap_values))[-5:][::-1]

            print(f"Top features pushing towards prediction:")
            for idx in top_indices:
                val = class_shap_values[idx]
                
                # Check bounds safety
                if idx < len(all_feature_names):
                    feature_name = all_feature_names[idx]
                    
                    # Only show if meaningful impact
                    if abs(val) > 0.001: 
                        print(f"  • {feature_name:<20} (Impact: {val:+.4f})")
                else:
                    print(f"  • Feature[{idx}]       (Impact: {val:+.4f})")
                    
        except Exception as e:
            print(f"Could not generate explanation: {e}")
                
    print("-" * 60)