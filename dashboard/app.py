import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import shap
import warnings
from pathlib import Path

# Paths are relative to the repository root
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "final_for_modeling.csv"
OUT = ROOT / "outputs"
warnings.filterwarnings('ignore')

# page config - setting this up first so the layout looks clean
st.set_page_config(
    page_title="Hospitalization Intensity Dashboard",
    layout="wide",
    page_icon="🏥"
)

st.title("🏥 Predicting Hospitalization Intensity Among U.S. Adults")
st.markdown("**MEPS Panels 6-10 (2001-2006) | ALY 6980 Capstone | Sharan Venkatesh**")
st.markdown("---")

# sidebar with project overview
with st.sidebar:
    st.markdown("### Project Overview")
    st.markdown("""
    **Course:** ALY 6980 Capstone  
    **Student:** Sharan Venkatesh  
    **Term:** Winter 2026
    """)
    st.markdown("---")
    st.markdown("### Research Question")
    st.markdown("Can hospitalization intensity be predicted among U.S. adults using demographic, socioeconomic, and health utilization data?")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    - **Source:** MEPS Panels 6-10
    - **Years:** 2001-2006
    - **Observations:** 53,140 adults (50,733 with complete data used for modeling)
    - **Outcome:** None / Single / Multiple hospitalizations
    """)
    st.markdown("---")
    st.markdown("### Models Built")
    st.markdown("""
    - Logistic Regression (Elastic Net)
    - Random Forest
    - XGBoost
    """)
    st.markdown("---")
    st.markdown("### Key Finding")
    st.info("No single model dominates. Random Forest has the best overall discrimination (AUC 0.844), while Logistic Regression identifies the most high-risk patients (Multiple Recall 62.9%).")
    st.markdown("---")
    st.markdown("### Dashboard Sections")
    st.markdown("""
    1. Sample Profile
    2. Model Performance
    3. Confusion Matrices
    4. SHAP Feature Importance
    """)

# loading the data - fixing the None string issue that pandas misreads as NaN
@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df['hosp_class_label'] = df['hosp_class_label'].fillna('None')
    return df

# loading the saved logistic regression model
@st.cache_resource
def load_logreg():
    with open(OUT / "model1_logreg_final.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

df = load_data()
comparison_df = pd.read_csv(OUT / "final_model_comparison.csv")
logreg_artifacts = load_logreg()

st.success(f"Data loaded: {len(df):,} observations across MEPS Panels 6-10")

# ── SECTION 1: SAMPLE PROFILE ──
st.header("1. Sample Profile")
st.markdown("Understanding who is in the dataset before looking at model results.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Observations", "53,140")
col2.metric("MEPS Panels", "6 to 10 (2001-2006)")
col3.metric("Outcome Classes", "None / Single / Multiple")

# class distribution chart
class_counts = df['hosp_class_label'].value_counts().reindex(['None', 'Single', 'Multiple'])
class_pct = (class_counts / class_counts.sum() * 100).round(1)

fig1 = go.Figure(go.Bar(
    x=class_counts.index,
    y=class_counts.values,
    text=[f"{v:,}<br>({p}%)" for v, p in zip(class_counts.values, class_pct)],
    textposition='outside',
    marker_color=['#2196F3', '#FF9800', '#F44336']
))
fig1.update_layout(
    title='Hospitalization Class Distribution',
    xaxis_title='Hospitalization Class',
    yaxis_title='Number of Adults',
    plot_bgcolor='white',
    yaxis=dict(gridcolor='lightgrey'),
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

# age and gender side by side
fig2 = make_subplots(rows=1, cols=2,
                     subplot_titles=('Age Distribution by Class',
                                     'Gender Distribution by Class'))
colors = {'None': '#2196F3', 'Single': '#FF9800', 'Multiple': '#F44336'}

for cls in ['None', 'Single', 'Multiple']:
    subset = df[df['hosp_class_label'] == cls]['age']
    fig2.add_trace(
        go.Histogram(x=subset, name=cls, opacity=0.7,
                     marker_color=colors[cls], xbins=dict(size=5),
                     legendgroup=cls),
        row=1, col=1
    )

gender_grp = df.groupby(['hosp_class_label', 'female']).size().reset_index(name='count')
gender_grp['gender'] = gender_grp['female'].map({1: 'Female', 0: 'Male'})
gender_grp['hosp_class_label'] = pd.Categorical(
    gender_grp['hosp_class_label'],
    categories=['None', 'Single', 'Multiple'], ordered=True)
gender_grp = gender_grp.sort_values('hosp_class_label')

for gender, color in [('Female', '#E91E63'), ('Male', '#3F51B5')]:
    subset = gender_grp[gender_grp['gender'] == gender]
    fig2.add_trace(
        go.Bar(x=subset['hosp_class_label'], y=subset['count'],
               name=gender, marker_color=color, legendgroup=gender),
        row=1, col=2
    )

fig2.update_layout(height=400, plot_bgcolor='white', barmode='group')
fig2.update_xaxes(gridcolor='lightgrey')
fig2.update_yaxes(gridcolor='lightgrey')
st.plotly_chart(fig2, use_container_width=True)

# ── SECTION 2: MODEL PERFORMANCE ──
st.markdown("---")
st.header("2. Model Performance Comparison")
st.markdown("All three models use class weights to handle the skewed outcome distribution. Total healthcare expenditure was excluded as a predictor after confirming it caused data leakage.")

# summary metrics table at the top so it's easy to read
st.subheader("Overall Metrics")
st.dataframe(
    comparison_df[['Model', 'Accuracy', 'Macro F1', 'Macro AUC', 
                   'Multiple Recall', 'Single Recall']].style.highlight_max(
        subset=['Macro AUC', 'Macro F1', 'Multiple Recall', 'Single Recall'],
        color='lightgreen'
    ).format(precision=3),
    use_container_width=True
)

# bar charts for the four key metrics
metrics = ['Macro AUC', 'Macro F1', 'Multiple Recall', 'Single Recall']
fig4 = make_subplots(rows=1, cols=4, subplot_titles=metrics)

model_colors = ['#4CAF50', '#2196F3', '#FF9800']

for i, metric in enumerate(metrics):
    values = comparison_df[metric].tolist()
    fig4.add_trace(
        go.Bar(
            x=comparison_df['Model'].tolist(),
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            marker_color=model_colors,
            showlegend=False
        ),
        row=1, col=i+1
    )

fig4.update_layout(
    height=450,
    title_text='Key Metrics Across All Three Models',
    plot_bgcolor='white'
)
fig4.update_yaxes(range=[0, 1.1], gridcolor='lightgrey')
fig4.update_xaxes(tickangle=15)
st.plotly_chart(fig4, use_container_width=True)

# callout boxes summarizing the key takeaway from each model
st.subheader("Key Takeaways")
c1, c2, c3 = st.columns(3)
c1.info("**Logistic Regression** \n\n Best at catching high-risk patients. Multiple Recall of 62.9% means it identified 6 out of 10 patients who would have multiple hospitalizations.")
c2.success("**Random Forest** \n\n Best overall risk discrimination. Macro AUC of 0.844 means it ranks patients by hospitalization risk better than the other two models.")
c3.warning("**XGBoost** \n\n Best at identifying single hospitalization patients. Single Recall of 54.5% makes it useful for detecting planned or procedural admissions.")

# ── SECTION 3: CONFUSION MATRICES ──
st.markdown("---")
st.header("3. Confusion Matrices")
st.markdown("Generating confusion matrices from actual model predictions on the test set.")

# preparing features the same way as the original modeling notebook
drop_cols = ['DUPERSID', 'panel', 'hospitalized', 'hosp_class',
             'hosp_class_label', 'race', 'total_exp']

df_model = df.copy()

# ordinal encoding to match original notebook
health_map = {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
poverty_map = {'High Income': 1, 'Middle Income': 2, 'Low Income': 3,
               'Near Poor': 4, 'Poor': 5}

df_model['health_status_num'] = df_model['health_status'].map(health_map)
df_model['mental_health_num'] = df_model['mental_health'].map(health_map)
df_model['poverty_cat_num'] = df_model['poverty_cat'].map(poverty_map)

# dropping rows where mapping produced NaN and any other missing values
df_model = df_model.dropna(subset=['health_status_num', 'mental_health_num', 'poverty_cat_num'])
df_model = df_model.dropna()

# drop text and unused columns
extra_drop = ['health_status', 'mental_health', 'poverty_cat',
              'race', 'region', 'insurance']
all_drop = drop_cols + extra_drop
feature_cols = [c for c in df_model.columns if c not in all_drop]

X = df_model[feature_cols].copy()
y = df_model['hosp_class_label'].copy()

# same 80/20 stratified split with same random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scaling numerical columns
cols_to_scale = ['age', 'er_visits', 'num_conditions',
                 'num_unique_meds', 'health_status_num',
                 'mental_health_num', 'poverty_cat_num']
cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

@st.cache_resource
def train_all_models(_X_train, _y_train):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # logistic regression with elastic net - same settings as modeling notebook
    logreg = LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, C=1.0,
        solver='saga', max_iter=5000,
        class_weight='balanced', random_state=42
    )
    logreg.fit(_X_train, _y_train)

    # random forest - same hyperparameters as modeling notebook
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_leaf=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(_X_train, _y_train)

    # xgboost needs numeric labels so encoding them
    le = LabelEncoder()
    y_encoded = le.fit_transform(_y_train)
    class_counts = pd.Series(_y_train).value_counts()
    total = len(_y_train)
    sample_weights = pd.Series(_y_train).map(
        lambda c: total / (3 * class_counts[c])
    ).values

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss'
    )
    xgb.fit(_X_train, y_encoded, sample_weight=sample_weights)

    return logreg, rf, xgb, le

with st.spinner("Training all three models... this takes about a minute"):
    logreg, rf, xgb, le = train_all_models(X_train_scaled, y_train)

st.success("Models trained. Generating confusion matrices.")

# generating predictions and plotting confusion matrices
labels = ['None', 'Single', 'Multiple']
fig5 = make_subplots(rows=1, cols=3,
                     subplot_titles=['Logistic Regression', 'Random Forest', 'XGBoost'])

# logistic regression and random forest predict string labels directly
for i, (name, model) in enumerate([('Logistic Regression', logreg), ('Random Forest', rf)]):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    z_text = [[f"{val:.1%}" for val in row] for row in cm]
    fig5.add_trace(
        go.Heatmap(z=cm, x=labels, y=labels, text=z_text,
                   texttemplate="%{text}", colorscale='Blues',
                   showscale=False, zmin=0, zmax=1),
        row=1, col=i+1
    )

# xgboost predicts numeric labels so decoding back to strings
y_pred_xgb = le.inverse_transform(xgb.predict(X_test_scaled))
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=labels, normalize='true')
z_text_xgb = [[f"{val:.1%}" for val in row] for row in cm_xgb]
fig5.add_trace(
    go.Heatmap(z=cm_xgb, x=labels, y=labels, text=z_text_xgb,
               texttemplate="%{text}", colorscale='Blues',
               showscale=True, zmin=0, zmax=1),
    row=1, col=3
)

fig5.update_layout(
    title_text='Normalized Confusion Matrices (rows = true class, columns = predicted)',
    height=450, plot_bgcolor='white'
)
fig5.update_xaxes(title_text='Predicted')
fig5.update_yaxes(title_text='Actual')
st.plotly_chart(fig5, use_container_width=True)

# ── SECTION 4: SHAP FEATURE IMPORTANCE ──
st.markdown("---")
st.header("4. SHAP Feature Importance")
st.markdown("SHAP values calculated on a subsample of 500 test observations. Showing top 15 features per class for Random Forest and XGBoost.")

@st.cache_resource
def get_shap_per_class(_rf_model, _xgb_model, _X_test):
    X_sample = _X_test.sample(500, random_state=42)

    # random forest SHAP - one array per class
    rf_explainer = shap.TreeExplainer(_rf_model)
    rf_shap_values = rf_explainer.shap_values(X_sample)

    # xgboost SHAP
    xgb_explainer = shap.TreeExplainer(_xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(X_sample)

    return rf_shap_values, xgb_shap_values, X_sample.columns.tolist()

with st.spinner("Calculating SHAP values... this takes about 30 seconds"):
    rf_shap, xgb_shap, feature_names = get_shap_per_class(rf, xgb, X_test_scaled)

# showing classes in natural order: None, Single, Multiple
display_classes = ['None', 'Single', 'Multiple']

class_colors = {
    'Multiple': '#F44336',
    'Single': '#FF9800',
    'None': '#4CAF50'
}

top_n = 15

# ── Random Forest per-class SHAP ──
st.subheader("Random Forest - Top 15 Features by SHAP Importance (Per Class)")

rf_fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Random Forest: {c}" for c in display_classes]
)

for i, cls in enumerate(display_classes):
    # get the index this class has in the model's original class order
    orig_idx = rf.classes_.tolist().index(cls)

    if isinstance(rf_shap, list):
        shap_vals = rf_shap[orig_idx]
    else:
        shap_vals = rf_shap[:, :, orig_idx]

    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-top_n:]
    top_features = [feature_names[j] for j in top_idx]
    top_values = importance[top_idx]

    rf_fig.add_trace(
        go.Bar(
            x=top_values,
            y=top_features,
            orientation='h',
            marker_color=class_colors.get(cls, '#888888'),
            showlegend=False
        ),
        row=1, col=i+1
    )

rf_fig.update_layout(
    height=500,
    plot_bgcolor='white',
    title_text='Random Forest - Top 15 Features by SHAP Importance'
)
rf_fig.update_xaxes(title_text='Mean |SHAP Value|', gridcolor='lightgrey')
rf_fig.update_yaxes(autorange='reversed')
st.plotly_chart(rf_fig, use_container_width=True)

# ── XGBoost per-class SHAP ──
st.subheader("XGBoost - Top 15 Features by SHAP Importance (Per Class)")

xgb_fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"XGBoost: {c}" for c in display_classes]
)

for i, cls in enumerate(display_classes):
    # get the index this class has in the label encoder's original order
    orig_idx = le.classes_.tolist().index(cls)

    if isinstance(xgb_shap, list):
        shap_vals = xgb_shap[orig_idx]
    else:
        shap_vals = xgb_shap[:, :, orig_idx]

    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-top_n:]
    top_features = [feature_names[j] for j in top_idx]
    top_values = importance[top_idx]

    xgb_fig.add_trace(
        go.Bar(
            x=top_values,
            y=top_features,
            orientation='h',
            marker_color=class_colors.get(cls, '#888888'),
            showlegend=False
        ),
        row=1, col=i+1
    )

xgb_fig.update_layout(
    height=500,
    plot_bgcolor='white',
    title_text='XGBoost - Top 15 Features by SHAP Importance'
)
xgb_fig.update_xaxes(title_text='Mean |SHAP Value|', gridcolor='lightgrey')
xgb_fig.update_yaxes(autorange='reversed')
st.plotly_chart(xgb_fig, use_container_width=True)

st.markdown("**Key finding:** ER visits and medication count are the top predictors across both models and all three classes, consistent with the clinical expectation that healthcare utilization patterns are the strongest signals of future hospitalization risk.")