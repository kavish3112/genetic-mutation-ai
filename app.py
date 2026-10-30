"""Flask web app + JSON API for an educational ClinVar variant classifier.
Not for clinical use. ClinVar data is downloaded at runtime.
"""
import io
import gzip
import re
import threading
import urllib.request

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

app = Flask(__name__)
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
GENES = ["BRCA1", "TP53", "CFTR", "APC", "MTHFR", "PMM2"]
STATE = {"raw": None, "gene": None, "model": None, "metrics": None,
         "counts": None, "error": None, "loading": False}
LOCK = threading.Lock()

def load_clinvar():
    req = urllib.request.Request(CLINVAR_URL, headers={"User-Agent": "GeneticMutationStudentProject/1.1"})
    with urllib.request.urlopen(req, timeout=240) as response:
        compressed = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
        return pd.read_csv(gz, sep="\t", low_memory=False, dtype=str)

def first_existing(df, names):
    return next((name for name in names if name in df.columns), None)

def infer_consequence(name, variant_type):
    text = str(name or "").lower()
    typ = str(variant_type or "").lower()
    protein = re.search(r"p\.([^\s,)]+)", text)
    ptext = protein.group(1).lower() if protein else ""
    if "fs" in ptext or "frameshift" in text: return "frameshift"
    if "ter" in ptext or "stop_gained" in text or re.search(r"p\.[a-z]{1,3}\d+\*", text): return "stop_gained"
    if "splice" in text or re.search(r"c\.[-*]?\d+[+-]\d+", text): return "splice_related"
    if re.fullmatch(r"[a-z]{1,3}\d+[a-z]{1,3}", ptext): return "missense_or_amino_acid_change"
    if "=" in ptext or re.search(r"c\.[-*]?\d+[a-z]>[a-z].*p\.[^ ]+=", text): return "synonymous_or_no_protein_change"
    if "deletion" in typ or "deletion" in text or "del" in typ: return "deletion"
    if "duplication" in typ or "duplication" in text or "dup" in typ: return "duplication"
    if "insertion" in typ or "insertion" in text or "ins" in typ: return "insertion"
    if "copy number" in typ or "cnv" in typ: return "copy_number_change"
    if "single nucleotide" in typ or "snv" in typ: return "single_nucleotide_other"
    return "other_or_unknown"

def extract_number(pattern, text):
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def prepare_gene_data(raw, gene):
    gene_col = first_existing(raw, ["GeneSymbol", "Gene"])
    sig_col = first_existing(raw, ["ClinicalSignificance", "Clinical significance"])
    if gene_col is None or sig_col is None:
        raise ValueError("Could not find expected gene/clinical-significance columns in ClinVar data.")
    df = raw[raw[gene_col].fillna("").str.split(r"[,;|]").apply(
        lambda values: gene.upper() in [v.strip().upper() for v in values])].copy()
    if df.empty: raise ValueError(f"No ClinVar records found for {gene}.")
    assembly_col = first_existing(df, ["Assembly"])
    if assembly_col:
        grch38 = df[assembly_col].fillna("").str.contains("GRCh38", case=False)
        if grch38.any(): df = df[grch38].copy()
    sig = df[sig_col].fillna("").str.lower()
    excluded = sig.str.contains(r"conflict|uncertain|risk factor|association|protective|drug response|not provided|other", regex=True)
    benign = sig.str.contains("benign", regex=False)
    pathogenic = sig.str.contains("pathogenic", regex=False)
    df = df[(benign & ~pathogenic & ~excluded) | (pathogenic & ~benign & ~excluded)].copy()
    if df.empty: raise ValueError(f"No unambiguous Benign/Pathogenic records remain for {gene}.")
    sig = df[sig_col].fillna("").str.lower()
    df["Label"] = np.where(sig.str.contains("pathogenic", regex=False), "Pathogenic", "Benign")
    vid = first_existing(df, ["VariationID"])
    if vid: df = df.drop_duplicates(subset=[vid], keep="first").copy()
    pos = first_existing(df, ["PositionVCF", "Start", "Position"])
    ref = first_existing(df, ["ReferenceAlleleVCF", "ReferenceAllele", "Reference allele"])
    alt = first_existing(df, ["AlternateAlleleVCF", "AlternateAllele", "Alternate allele"])
    typ = first_existing(df, ["Type", "VariantType"])
    name = first_existing(df, ["Name"])
    chrom = first_existing(df, ["Chromosome"])
    df["position"] = pd.to_numeric(df[pos], errors="coerce") if pos else np.nan
    df["ref_length"] = df[ref].fillna("").str.len().astype(float) if ref else 0.0
    df["alt_length"] = df[alt].fillna("").str.len().astype(float) if alt else 0.0
    df["variant_type"] = df[typ].fillna("Unknown").astype(str) if typ else "Unknown"
    df["chromosome"] = df[chrom].fillna("Unknown").astype(str) if chrom else "Unknown"
    df["variant_name"] = df[name].fillna("").astype(str) if name else ""
    df["consequence_group"] = [infer_consequence(n, t) for n, t in zip(df["variant_name"], df["variant_type"])]
    df["protein_position"] = df["variant_name"].apply(lambda x: extract_number(r"p\.(?:[A-Za-z]{1,3})(\d+)", x))
    df["cdna_position"] = df["variant_name"].apply(lambda x: extract_number(r"c\.[-*]?(\d+)", x))
    valid = df["position"].dropna()
    if len(valid) > 1 and valid.max() > valid.min():
        df["relative_gene_position"] = (df["position"] - valid.min()) / (valid.max() - valid.min())
    else: df["relative_gene_position"] = 0.5
    df["evaluation_group"] = np.where(df["protein_position"].notna(),
        "protein_" + df["protein_position"].fillna(-1).astype(str),
        "genomic_" + df["position"].fillna(-1).astype(str))
    features = ["chromosome", "variant_type", "consequence_group", "ref_length", "alt_length",
                "relative_gene_position", "protein_position", "cdna_position"]
    model_df = df[features + ["Label", "evaluation_group"]].replace([np.inf, -np.inf], np.nan)
    return df, model_df

def make_pipeline():
    nums = ["ref_length", "alt_length", "relative_gene_position", "protein_position", "cdna_position"]
    cats = ["chromosome", "variant_type", "consequence_group"]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ])
    return Pipeline([("preprocess", pre), ("model", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))])

def train_model(model_df):
    X = model_df.drop(columns=["Label", "evaluation_group"])
    y = model_df["Label"].reset_index(drop=True)
    groups = model_df["evaluation_group"].reset_index(drop=True)
    if y.nunique() < 2: raise ValueError("This gene needs both Benign and Pathogenic examples.")
    if y.value_counts().min() < 4: raise ValueError("Too few examples in one class. Try BRCA1, TP53, or CFTR.")
    splitter = GroupShuffleSplit(n_splits=60, test_size=0.25, random_state=42)
    chosen = None
    for tr, te in splitter.split(X, y, groups):
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            chosen = tr, te
            break
    if chosen is None: raise ValueError("Could not create a grouped train/test split. Try another gene.")
    tr, te = chosen
    model = make_pipeline()
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    prob = model.predict_proba(X.iloc[te])
    classes = list(model.named_steps["model"].classes_)
    auc = roc_auc_score((y.iloc[te] == "Pathogenic").astype(int), prob[:, classes.index("Pathogenic")])
    metrics = {
        "Accuracy": round(float(accuracy_score(y.iloc[te], pred)), 3),
        "Balanced accuracy": round(float(balanced_accuracy_score(y.iloc[te], pred)), 3),
        "Precision (Pathogenic)": round(float(precision_score(y.iloc[te], pred, pos_label="Pathogenic", zero_division=0)), 3),
        "Recall (Pathogenic)": round(float(recall_score(y.iloc[te], pred, pos_label="Pathogenic", zero_division=0)), 3),
        "F1 (Pathogenic)": round(float(f1_score(y.iloc[te], pred, pos_label="Pathogenic", zero_division=0)), 3),
        "ROC-AUC": round(float(auc), 3)
    }
    return model, metrics

@app.get("/")
def home():
    return render_template("index.html", genes=GENES)

@app.get("/api/status")
def status():
    return jsonify({"trained": STATE["model"] is not None, "gene": STATE["gene"],
                    "loading": STATE["loading"], "error": STATE["error"],
                    "metrics": STATE["metrics"], "counts": STATE["counts"]})

@app.post("/api/train")
def api_train():
    gene = str(request.json.get("gene", "BRCA1") if request.is_json else request.form.get("gene", "BRCA1")).upper()
    if gene not in GENES: return jsonify({"error": "Gene not supported in this prototype."}), 400
    with LOCK:
        if STATE["loading"]: return jsonify({"error": "Training is already running. Please wait."}), 409
        STATE["loading"] = True
        STATE["error"] = None
    try:
        if STATE["raw"] is None: STATE["raw"] = load_clinvar()
        original, model_df = prepare_gene_data(STATE["raw"], gene)
        model, metrics = train_model(model_df)
        STATE.update({"gene": gene, "model": model, "metrics": metrics,
                      "counts": {"usable_variants": int(len(model_df)),
                                 "benign": int((model_df["Label"] == "Benign").sum()),
                                 "pathogenic": int((model_df["Label"] == "Pathogenic").sum())},
                      "error": None})
        return jsonify({"message": f"Model trained for {gene}.", "gene": gene,
                        "metrics": metrics, "counts": STATE["counts"]})
    except Exception as exc:
        STATE["error"] = str(exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        STATE["loading"] = False

@app.post("/api/predict")
def api_predict():
    if STATE["model"] is None:
        return jsonify({"error": "Train a model first."}), 400
    data = request.get_json(silent=True) or {}
    try:
        sample = pd.DataFrame([{
            "chromosome": str(data.get("chromosome", "Unknown")),
            "variant_type": str(data.get("variant_type", "single nucleotide variant")),
            "consequence_group": str(data.get("consequence_group", "missense_or_amino_acid_change")),
            "ref_length": float(data.get("ref_length", 1)),
            "alt_length": float(data.get("alt_length", 1)),
            "relative_gene_position": float(data.get("relative_gene_position", 0.5)),
            "protein_position": float(data["protein_position"]) if data.get("protein_position") not in (None, "", 0, "0") else np.nan,
            "cdna_position": float(data["cdna_position"]) if data.get("cdna_position") not in (None, "", 0, "0") else np.nan,
        }])
        model = STATE["model"]
        prediction = str(model.predict(sample)[0])
        classes = list(model.named_steps["model"].classes_)
        scores = {label: round(float(score), 4) for label, score in zip(classes, model.predict_proba(sample)[0])}
        return jsonify({"gene": STATE["gene"], "prediction": prediction, "model_scores": scores,
                        "notice": "Educational research output only; not a clinical interpretation."})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
