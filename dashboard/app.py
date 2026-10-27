"""
Hospitalization Intensity Dashboard
Run from the repository root:  python3 -m streamlit run dashboard/app.py

The models here are rebuilt with exactly the same features, split and settings as
02_logistic_regression.ipynb and 03_random_forest_xgboost.ipynb, so every number on
the page comes from one consistent set of predictions.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "final_for_modeling.csv"

CLASSES = ["None", "Single", "Multiple"]
CLASS_COLORS = {"None": "#0071e3", "Single": "#ff9f0a", "Multiple": "#ff375f"}
MODEL_COLORS = {"Logistic Regression": "#34c759", "Random Forest": "#0071e3", "XGBoost": "#ff9f0a"}
FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'

# Leakage test results from 02_logistic_regression.ipynb (Macro AUC with vs. without total spending)
LEAK = {"SMOTE": (0.923, 0.763), "Class weights": (0.938, 0.807)}

NICE = {
    "er_visits": "ER visits", "num_unique_meds": "Medications", "polypharmacy": "5+ medications",
    "health_status_num": "Self-rated health", "mental_health_num": "Mental health",
    "num_conditions": "Chronic conditions", "age": "Age", "female": "Female", "married": "Married",
    "employed": "Employed", "has_limitation": "Activity limitation", "poverty_cat_num": "Poverty level",
    "has_heart_disease": "Heart disease", "has_hypertension": "Hypertension", "has_diabetes": "Diabetes",
    "has_copd": "COPD", "has_depression": "Depression", "has_arthritis": "Arthritis", "has_stroke": "Stroke",
    "insurance_Public": "Public insurance", "insurance_Uninsured": "Uninsured",
    "region_Northeast": "Northeast", "region_South": "South", "region_West": "West",
    "race_White": "White", "race_Other": "Other race",
}

st.set_page_config(page_title="Hospitalization Intensity", page_icon="🏥",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
html, body, .stApp, [class*="st-"], button, input {{ font-family: {FONT}; }}
.stApp {{ background: #f5f5f7; color: #1d1d1f; }}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ max-width: 1120px; padding: 0 2rem 6rem; }}

.nav {{ position: sticky; top: 0; z-index: 99; display: flex; justify-content: center; gap: 34px;
        padding: 14px 0; font-size: 12px; letter-spacing: .2px;
        background: rgba(245,245,247,.75); backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px); border-bottom: 1px solid rgba(0,0,0,.07); }}
.nav a {{ color: #1d1d1f !important; text-decoration: none; opacity: .8; }}
.nav a:hover {{ opacity: 1; }}

.hero {{ text-align: center; padding: 96px 0 44px; }}
.eyebrow {{ color: #bf4800; font-size: 17px; font-weight: 600; margin: 0 0 10px; }}
.hero h1 {{ font-size: clamp(44px, 6.5vw, 80px); font-weight: 700; letter-spacing: -.015em;
            line-height: 1.04; margin: 0; color: #1d1d1f; padding: 0; }}
.hero .lede {{ font-size: clamp(19px, 2vw, 24px); line-height: 1.4; color: #6e6e73; max-width: 720px;
               margin: 22px auto 0; font-weight: 400; }}

.sec {{ text-align: center; padding: 110px 0 34px; scroll-margin-top: 60px; }}
.sec .eyebrow {{ color: #6e6e73; font-size: 17px; }}
.sec h2 {{ font-size: clamp(34px, 4.6vw, 56px); font-weight: 700; letter-spacing: -.01em;
           line-height: 1.07; margin: 0; color: #1d1d1f; padding: 0; }}
.sec p.sub {{ font-size: 19px; line-height: 1.45; color: #6e6e73; max-width: 680px; margin: 18px auto 0; }}

.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
.tile {{ background: #fff; border-radius: 28px; padding: 34px 30px; box-shadow: 0 4px 24px rgba(0,0,0,.05);
         display: flex; flex-direction: column; justify-content: flex-start; }}
.tile > div, .tile p {{ margin: 0 !important; }}
.tile .k {{ font-size: 14px; font-weight: 600; color: #6e6e73; }}
.tile .v {{ font-size: clamp(40px, 4.6vw, 58px); font-weight: 700; letter-spacing: -.02em; line-height: 1.05; margin-top: 8px; }}
.tile .d {{ font-size: 15px; line-height: 1.45; color: #6e6e73; margin-top: 10px; }}
.tile.dark {{ background: #1d1d1f; }}
.tile.dark .k, .tile.dark .d {{ color: #a1a1a6; }}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #fff; border: none !important; outline: none !important; border-radius: 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,.05); padding: 18px 24px !important; overflow: hidden; }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ border: none !important; }}
[data-testid="stElementContainer"]:has([data-testid="stRadio"]), [data-testid="stRadio"], .stRadio {{
    display: flex !important; justify-content: center !important; width: 100% !important; }}
[data-testid="stRadio"] > div {{ margin: 0 auto; }}
div[role="radiogroup"] {{ justify-content: center; gap: 4px; background: #e8e8ed; border-radius: 980px;
                          padding: 4px; width: fit-content; margin: 0 auto 10px; }}
div[role="radiogroup"] label {{ padding: 6px 20px; border-radius: 980px; margin: 0 !important; cursor: pointer; }}
div[role="radiogroup"] label:has(input:checked) {{ background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
div[role="radiogroup"] label > div:first-child {{ display: none; }}

.note {{ text-align: center !important; color: #86868b; font-size: 13px; line-height: 1.5; max-width: 760px;
         margin: 14px auto 0 !important; display: block; }}
.foot {{ text-align: center; color: #86868b; font-size: 12px; line-height: 1.6; padding: 40px 0 0;
         border-top: 1px solid #d2d2d7; margin-top: 110px; }}
</style>
""", unsafe_allow_html=True)


def section(anchor, eyebrow, title, sub=""):
    st.markdown(f'<div class="sec" id="{anchor}"><p class="eyebrow">{eyebrow}</p><h2>{title}</h2>'
                + (f'<p class="sub">{sub}</p>' if sub else "") + "</div>", unsafe_allow_html=True)


def tiles(items):
    """items: (label, big value, detail, value colour, extra css class)"""
    html = '<div class="tiles">' + "".join(
        f'<div class="tile {cls}"><div class="k">{k}</div><div class="v" style="color:{c}">{v}</div>'
        f'<div class="d">{d}</div></div>' for k, v, d, c, cls in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(height=height, font=dict(family=FONT, size=13, color="#1d1d1f"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(font_family=FONT),
                      legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d2d2d7")
    fig.update_yaxes(gridcolor="#ececf0", zeroline=False)
    for a in fig.layout.annotations:
        a.font = dict(family=FONT, size=15, color="#1d1d1f")
    return fig


def chart(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------- data + models (same as the notebooks)
@st.cache_data
def load_data():
    # keep_default_na=False stops pandas reading the class label "None" as missing
    df = pd.read_csv(DATA, keep_default_na=False, na_values=["", "NA", "NaN"])
    df["hosp_class_label"] = df["hosp_class"].map({0: "None", 1: "Single", 2: "Multiple"})
    return df


@st.cache_data
def prepare(df):
    nominal = ["race", "region", "insurance"]
    ordinal = ["health_status", "mental_health", "poverty_cat"]
    binary = ["female", "married", "employed", "has_limitation", "has_diabetes", "has_hypertension",
              "has_heart_disease", "has_copd", "has_depression", "has_arthritis", "has_stroke", "polypharmacy"]
    numeric = ["age", "er_visits", "num_conditions", "num_unique_meds"]
    health = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    d = df.copy()
    d["health_status_num"] = d["health_status"].map(health)
    d["mental_health_num"] = d["mental_health"].map(health)
    d["poverty_cat_num"] = d["poverty_cat"].map({"High Income": 1, "Middle Income": 2, "Low Income": 3,
                                                 "Near Poor": 4, "Poor": 5})
    d = pd.get_dummies(d, columns=nominal, drop_first=True, dtype=int)
    dummies = [c for c in d.columns if any(c.startswith(n + "_") for n in nominal)]
    feats = binary + numeric + ["health_status_num", "mental_health_num", "poverty_cat_num"] + dummies
    d = d.dropna(subset=feats)
    drop = ["DUPERSID", "panel", "hospitalized", "hosp_class", "hosp_class_label", "total_exp"] + ordinal
    X = d.drop(columns=[c for c in drop if c in d.columns])
    y = d["hosp_class_label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), len(d)


SCALED = ["age", "er_visits", "num_conditions", "num_unique_meds",
          "health_status_num", "mental_health_num", "poverty_cat_num"]


@st.cache_resource
def train(_X_tr, _y_tr, _X_te):
    # Logistic regression: scaled features, class weights (notebook 2, configuration 4)
    scaler = StandardScaler().fit(_X_tr[SCALED])
    Xs_tr, Xs_te = _X_tr.copy(), _X_te.copy()
    Xs_tr[SCALED] = scaler.transform(_X_tr[SCALED])
    Xs_te[SCALED] = scaler.transform(_X_te[SCALED])
    lr = LogisticRegression(penalty="elasticnet", solver="saga", C=1.0, l1_ratio=0.5, max_iter=5000,
                            random_state=42, class_weight="balanced", n_jobs=-1).fit(Xs_tr, _y_tr)
    # Random Forest and XGBoost: unscaled features, same settings as notebook 3
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(_X_tr, _y_tr)
    lab = {"None": 0, "Single": 1, "Multiple": 2}
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, min_child_weight=20, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    xgb.fit(_X_tr, _y_tr.map(lab), sample_weight=_y_tr.map({"None": 0.36, "Single": 5.11, "Multiple": 20.28}))

    preds = {
        "Logistic Regression": (lr.predict(Xs_te),
                                pd.DataFrame(lr.predict_proba(Xs_te), columns=lr.classes_)[CLASSES].values),
        "Random Forest": (rf.predict(_X_te),
                          pd.DataFrame(rf.predict_proba(_X_te), columns=rf.classes_)[CLASSES].values),
        "XGBoost": (np.array(CLASSES)[xgb.predict(_X_te)], xgb.predict_proba(_X_te)),
    }
    return rf, xgb, preds


@st.cache_resource
def shap_importance(_rf, _xgb, _X_te):
    np.random.seed(42)  # same 1,000-adult sample as notebook 3
    sample = _X_te.loc[np.random.choice(_X_te.index, size=1000, replace=False)]
    out = {}
    for name, model, order in [("Random Forest", _rf, list(_rf.classes_)), ("XGBoost", _xgb, CLASSES)]:
        vals = shap.TreeExplainer(model).shap_values(sample)
        for cls in CLASSES:
            k = order.index(cls)
            v = vals[k] if isinstance(vals, list) else vals[:, :, k]
            out[(name, cls)] = pd.Series(np.abs(v).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    return out


df = load_data()
(X_tr, X_te, y_tr, y_te), n_model = prepare(df)

with st.spinner("Training the three models. This takes about a minute the first time."):
    rf, xgb, preds = train(X_tr, y_tr, X_te)

rows = []
for name, (yp, proba) in preds.items():
    rec = recall_score(y_te, yp, labels=CLASSES, average=None)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, yp),
                 "Macro F1": f1_score(y_te, yp, average="macro"),
                 "Macro AUC": roc_auc_score(label_binarize(y_te, classes=CLASSES), proba,
                                            multi_class="ovr", average="macro"),
                 "Multiple Recall": rec[2], "Single Recall": rec[1]})
M = pd.DataFrame(rows).set_index("Model")
best_auc, best_mult, best_single = M["Macro AUC"].idxmax(), M["Multiple Recall"].idxmax(), M["Single Recall"].idxmax()

# ---------------------------------------------------------------- nav + hero
st.markdown('<div class="nav"><a href="#sample">Sample</a><a href="#leak">The leak</a>'
            '<a href="#models">Models</a><a href="#errors">Errors</a><a href="#drivers">Drivers</a></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <p class="eyebrow">MEPS Panels 6–10 · 2001–2006</p>
  <h1>Hospitalization intensity.<br>Predicted.</h1>
  <p class="lede">A three-class model that sorts adults into no, one, or repeat hospital stays,
  and the data leak it had to get past first.</p>
</div>""", unsafe_allow_html=True)

tiles([
    ("Adults modeled", f"{n_model:,}", f"With complete data, out of {len(df):,} MEPS adults.", "#1d1d1f", ""),
    ("Best ranking", f"{M.loc[best_auc, 'Macro AUC']:.3f}", f"Macro AUC · {best_auc}", "#0071e3", ""),
    ("Repeat stays caught", f"{M.loc[best_mult, 'Multiple Recall']:.0%}",
     f"Multiple-class recall · {best_mult}", "#ff375f", ""),
])

# ---------------------------------------------------------------- 1. sample
counts = df["hosp_class_label"].value_counts().reindex(CLASSES)
pct = counts / counts.sum()
section("sample", "The sample", "Most adults never go in.",
        "Fewer than 1 in 10 had a hospital stay, and fewer than 2 in 100 had more than one. "
        "That imbalance shapes every modeling choice that follows.")
tiles([(c, f"{pct[c]:.1%}", f"{counts[c]:,} adults", CLASS_COLORS[c], "") for c in CLASSES])

st.write("")
with st.container(border=True):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Age, share within each class", "Share who are female"),
                        horizontal_spacing=0.1)
    bins = np.arange(15, 95, 5)
    for c in CLASSES:
        ages = df.loc[df.hosp_class_label == c, "age"].dropna()
        h, _ = np.histogram(ages, bins=bins)
        fig.add_trace(go.Scatter(x=bins[:-1] + 2.5, y=h / h.sum() * 100, name=c, mode="lines+markers",
                                 line=dict(color=CLASS_COLORS[c], width=3, shape="spline"),
                                 marker=dict(size=6), hovertemplate="Age %{x:.0f}: %{y:.1f}%<extra>" + c + "</extra>"),
                      1, 1)
    fem = df.groupby("hosp_class_label")["female"].mean().reindex(CLASSES)
    fig.add_trace(go.Bar(x=CLASSES, y=fem.values, marker_color=[CLASS_COLORS[c] for c in CLASSES],
                         text=[f"{v:.0%}" for v in fem], textposition="outside", showlegend=False,
                         marker_line_width=0), 1, 2)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, fem.max() * 1.25], row=1, col=2)
    chart(style(fig, 400))
overall_fem = df["female"].mean()
st.markdown(f'<p class="note">Repeat stays skew older. And {fem["Single"]:.0%} of single-stay adults are female, '
            f'versus {overall_fem:.0%} overall, which likely reflects childbirth admissions (MEPS counts them as '
            f'hospital stays). That is also why "Female" ranks highly for the Single class below.</p>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- 2. the leak
(s_with, s_without), (w_with, w_without) = LEAK["SMOTE"], LEAK["Class weights"]
section("leak", "The catch", "The best predictor was cheating.",
        "Total healthcare spending looked like a strong signal, but it already includes the cost of the "
        "hospital stays being predicted. Removing it showed how much of the accuracy was borrowed.")
tiles([
    ("With spending", f"{w_with:.3f}", "Macro AUC, logistic regression. Looks great, but it is leakage.",
     "#86868b", ""),
    ("Without spending", f"{w_without:.3f}",
     f"The honest score, down {w_with - w_without:.2f} (SMOTE version: down {s_with - s_without:.2f}).",
     "#f5f5f7", "dark"),
    ("Every model below", "Leak-free", "Class weights for the imbalance, and no spending variable.",
     "#34c759", ""),
])

# ---------------------------------------------------------------- 3. models
section("models", "Three models", "No single winner.",
        f"{best_auc} ranks risk best. {best_mult} catches the most repeat stays. "
        f"{best_single} is strongest on single stays.")
tiles([
    (name, f"{M.loc[name, 'Macro AUC']:.3f}",
     f"Macro AUC<br>Repeat-stay recall {M.loc[name, 'Multiple Recall']:.0%} · "
     f"single-stay recall {M.loc[name, 'Single Recall']:.0%}", MODEL_COLORS[name], "")
    for name in M.index
])
st.write("")
with st.container(border=True):
    cols = ["Macro AUC", "Macro F1", "Multiple Recall", "Single Recall"]
    fig = go.Figure()
    for name in M.index:
        fig.add_trace(go.Bar(name=name, x=cols, y=M.loc[name, cols], marker_color=MODEL_COLORS[name],
                             text=[f"{v:.2f}" for v in M.loc[name, cols]], textposition="outside",
                             marker_line_width=0))
    fig.update_layout(barmode="group", bargap=.28)
    fig.update_yaxes(range=[0, 1.05])
    chart(style(fig, 420))
st.markdown('<p class="note">Models are retrained here with the notebooks\' exact features, split and settings. '
            'Results can differ from the notebook outputs by about 1–2 points because of library versions.</p>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. errors
section("errors", "Where they miss", "The rare classes are the hard part.",
        "Each row is the true class, and each cell shows where those adults were predicted to land. "
        "The diagonal is what each model gets right. Most errors are rare cases pushed toward a neighboring class.")
with st.container(border=True):
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(preds), horizontal_spacing=0.1)
    for i, (name, (yp, _)) in enumerate(preds.items()):
        cm = confusion_matrix(y_te, yp, labels=CLASSES, normalize="true")
        fig.add_trace(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, text=[[f"{v:.0%}" for v in r] for r in cm],
                                 texttemplate="%{text}", textfont=dict(size=15), zmin=0, zmax=1,
                                 colorscale=[[0, "#f2f7fd"], [1, "#0071e3"]], showscale=False,
                                 xgap=4, ygap=4), 1, i + 1)
    fig.update_xaxes(title_text="Predicted", showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    style(fig, 420).update_layout(margin=dict(l=70, r=10, t=50, b=10))
    chart(fig)

# ---------------------------------------------------------------- 5. drivers
with st.spinner("Calculating SHAP values..."):
    imp = shap_importance(rf, xgb, X_te)

core = ["er_visits", "num_unique_meds"]
in_top5 = all(f in imp[k].index[:5] for k in imp for f in core)
section("drivers", "What drives risk", "Use of care says the most.",
        ("ER visits and medication count rank in the top five for both models and all three classes. "
         if in_top5 else "Healthcare use, age and self-rated health carry most of the signal. ")
        + "They are measured in the same year as the outcome, so they describe risk rather than forecast it.")
cls = st.radio("Class", CLASSES, index=2, horizontal=True, label_visibility="collapsed")
with st.container(border=True):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Random Forest", "XGBoost"), horizontal_spacing=0.22)
    for i, name in enumerate(["Random Forest", "XGBoost"]):
        s = imp[(name, cls)].head(10)[::-1]
        fig.add_trace(go.Bar(x=s.values, y=[NICE.get(f, f) for f in s.index], orientation="h",
                             marker_color=CLASS_COLORS[cls], marker_line_width=0, showlegend=False), 1, i + 1)
    fig.update_xaxes(title_text="Mean |SHAP value|", showgrid=True, gridcolor="#ececf0")
    fig.update_yaxes(showgrid=False)
    chart(style(fig, 460))
st.markdown(f'<p style="text-align:center;color:#86868b;font-size:13px">Top 10 features for the selected class. '
            f'SHAP on the same 1,000 test adults as notebook 3, out of {len(y_te):,} in the test set.</p>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- footer
st.markdown("""
<div class="foot">
  Sharan Venkatesh · Northeastern University MPS Analytics capstone (ALY 6980), Winter 2026<br>
  Data: Medical Expenditure Panel Survey (AHRQ), Panels 6–10. Survey weights not applied; results describe the MEPS sample.
</div>""", unsafe_allow_html=True)
