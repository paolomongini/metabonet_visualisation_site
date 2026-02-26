import streamlit as st
import duckdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import timedelta
import io
from scipy.io import savemat
import gdown
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="CGM Explorer", page_icon="📈", layout="wide")

def inject_ga():
    try:
        with open("google_analytics.html", "r") as f:
            html_code = f.read()
            # Usiamo height=0 e lo posizioniamo in alto
            components.html(html_code, height=0)
    except FileNotFoundError:
        pass # Silenzioso per l'utente, o st.error per debug

inject_ga()


@st.cache_resource # Fondamentale: scarica il file una volta sola per sessione
def download_parquet():
    file_id = "1R-4JeAUCVm_0Xx8J4VgyC4jAly4G9rEc"
    url = f'https://drive.google.com/uc?id={file_id}'
    output = "metabonet_public_2025.parquet"
    
    if not os.path.exists(output):
        with st.spinner("Scaricamento del database (700MB+) in corso... attendi..."):
            gdown.download(url, output, quiet=False)
    return output

# Inizializza il database
PARQUET_PATH = download_parquet()

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.03em; }
  .block-container { padding: 2rem 2.5rem; }
  .metric-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
  .metric-val { font-size: 2rem; font-weight: 700; }
  .metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; }
  .nav-label { text-align: center; font-size: 1rem; font-weight: 700; color: #333; padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# CGM Explorer")
st.markdown("Continuous Glucose Monitoring — MetaboNet 2025")
st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def get_all_patients():
    con = duckdb.connect()
    return con.execute(f"SELECT DISTINCT id FROM '{PARQUET_PATH}' WHERE id IS NOT NULL ORDER BY id").df()['id'].tolist()

@st.cache_data
def get_patient_summary():
    """Pre-compute once: per-patient summary of available data types. Cached forever."""
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            id,
            MAX(age) as age,
            MAX(CASE WHEN CGM IS NOT NULL THEN 1 ELSE 0 END) as has_cgm,
            MAX(CASE WHEN (insulin IS NOT NULL AND insulin > 0)
                      OR  (bolus  IS NOT NULL AND bolus  > 0)
                      OR  (basal  IS NOT NULL AND basal  > 0) THEN 1 ELSE 0 END) as has_insulin,
            MAX(CASE WHEN carbs IS NOT NULL AND carbs > 0 THEN 1 ELSE 0 END) as has_carbs,
            MAX(CASE WHEN heartrate IS NOT NULL AND heartrate > 0 THEN 1 ELSE 0 END) as has_heartrate,
            MAX(CASE WHEN steps IS NOT NULL AND steps > 0 THEN 1 ELSE 0 END) as has_steps,
            MAX(CASE WHEN workout_duration IS NOT NULL AND workout_duration > 0 THEN 1 ELSE 0 END) as has_workout_duration,
            MAX(CASE WHEN workout_intensity IS NOT NULL AND workout_intensity > 0 THEN 1 ELSE 0 END) as has_workout_intensity,
            MAX(CASE WHEN TRY_CAST(workout_label AS DOUBLE) > 0 THEN 1 ELSE 0 END) as has_workout_label,
            MAX(treatment_group) as treatment_group
        FROM '{PARQUET_PATH}'
        WHERE id IS NOT NULL
        GROUP BY id
        ORDER BY id
    """).df()

def get_therapy_types():
    summary = get_patient_summary()
    return sorted(summary['treatment_group'].dropna().unique().tolist())

def get_filtered_patients(need_cgm, need_insulin, need_carbs,
                          need_heartrate, need_steps,
                          need_workout_duration, need_workout_intensity, need_workout_label,
                          therapy_filter, min_age, max_age, limit=None):
    summary = get_patient_summary()
    mask = pd.Series([True] * len(summary))
    if need_cgm:              mask &= (summary['has_cgm']              == 1)
    if need_insulin:          mask &= (summary['has_insulin']          == 1)
    if need_carbs:            mask &= (summary['has_carbs']            == 1)
    if need_heartrate:        mask &= (summary['has_heartrate']        == 1)
    if need_steps:            mask &= (summary['has_steps']            == 1)
    if need_workout_duration: mask &= (summary['has_workout_duration'] == 1)
    if need_workout_intensity:mask &= (summary['has_workout_intensity']== 1)
    if need_workout_label:    mask &= (summary['has_workout_label']    == 1)
    if therapy_filter:
        mask &= summary['treatment_group'].isin(therapy_filter)
    # Age filter (only when summary has age)
    if 'age' in summary.columns:
        if min_age is not None:
            mask &= (summary['age'].fillna(-1) >= min_age)
        if max_age is not None:
            mask &= (summary['age'].fillna(9999) <= max_age)
    result = summary[mask]['id'].tolist()
    if limit:
        result = result[:limit]
    return result

@st.cache_data
def get_patient_dates(patient_id):
    con = duckdb.connect()
    row = con.execute(f"SELECT MIN(date) as mn, MAX(date) as mx FROM '{PARQUET_PATH}' WHERE id = '{patient_id}'").df()
    return pd.Timestamp(row['mn'][0]), pd.Timestamp(row['mx'][0])

@st.cache_data
def get_window_data(patient_id, start, end):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT date, CGM, carbs, bolus, insulin, basal,
               heartrate, steps, workout_duration, workout_intensity, workout_label
        FROM '{PARQUET_PATH}'
        WHERE id = '{patient_id}'
          AND date >= '{start}' AND date < '{end}'
        ORDER BY date
    """).df()

@st.cache_data
def get_daily_coverage(patient_id, need_cgm, need_insulin, need_carbs,
                       need_heartrate=False, need_steps=False,
                       need_workout_duration=False, need_workout_intensity=False,
                       need_workout_label=False):
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            DATE_TRUNC('day', date) as day,
            MAX(CASE WHEN CGM IS NOT NULL THEN 1 ELSE 0 END) as has_cgm,
            MAX(CASE WHEN carbs IS NOT NULL AND carbs > 0 THEN 1 ELSE 0 END) as has_carbs,
            MAX(CASE WHEN (insulin IS NOT NULL AND insulin > 0)
                      OR (bolus IS NOT NULL AND bolus > 0)
                      OR (basal IS NOT NULL AND basal > 0) THEN 1 ELSE 0 END) as has_insulin,
            MAX(CASE WHEN heartrate IS NOT NULL AND heartrate > 0 THEN 1 ELSE 0 END) as has_heartrate,
            MAX(CASE WHEN steps IS NOT NULL AND steps > 0 THEN 1 ELSE 0 END) as has_steps,
            MAX(CASE WHEN workout_duration IS NOT NULL AND workout_duration > 0 THEN 1 ELSE 0 END) as has_workout_duration,
            MAX(CASE WHEN workout_intensity IS NOT NULL AND workout_intensity > 0 THEN 1 ELSE 0 END) as has_workout_intensity,
            MAX(CASE WHEN TRY_CAST(workout_label AS DOUBLE) > 0 THEN 1 ELSE 0 END) as has_workout_label
        FROM '{PARQUET_PATH}'
        WHERE id = '{patient_id}'
        GROUP BY DATE_TRUNC('day', date)
        ORDER BY day
    """).df()
    df['day'] = pd.to_datetime(df['day'])
    df['complete'] = True
    if need_cgm:              df['complete'] &= (df['has_cgm']              == 1)
    if need_carbs:            df['complete'] &= (df['has_carbs']            == 1)
    if need_insulin:          df['complete'] &= (df['has_insulin']          == 1)
    if need_heartrate:        df['complete'] &= (df['has_heartrate']        == 1)
    if need_steps:            df['complete'] &= (df['has_steps']            == 1)
    if need_workout_duration: df['complete'] &= (df['has_workout_duration'] == 1)
    if need_workout_intensity:df['complete'] &= (df['has_workout_intensity']== 1)
    if need_workout_label:    df['complete'] &= (df['has_workout_label']    == 1)
    return df

@st.cache_data
def get_patient_metadata(patient_id):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT 
            id,
            ANY_VALUE(age) as age,
            ANY_VALUE(gender) as gender,
            ANY_VALUE(ethnicity) as ethnicity,
            ANY_VALUE(weight) as weight,
            ANY_VALUE(height) as height,
            ANY_VALUE(treatment_group) as treatment_group,
            ANY_VALUE(age_of_diagnosis) as age_diagnosis,
            ANY_VALUE(cgm_device) as cgm_device
        FROM '{PARQUET_PATH}'
        WHERE id = '{patient_id}'
        GROUP BY id
    """).df().iloc[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Patient Selection")
    mode = st.radio("Mode", ["Manual", "Filter by data"], horizontal=True)

    therapy_filter = []
    # defaults for new filters
    need_heartrate = need_steps = need_workout_duration = need_workout_intensity = need_workout_label = False
    min_age = max_age = None

    if mode == "Manual":
        need_cgm = need_insulin = need_carbs = False
        patients = get_all_patients()
        selected = st.selectbox("Patient ID", ["— select —"] + patients)
    else:
        st.markdown("**Show only patients with:**")
        need_cgm              = st.checkbox("CGM data",              value=True)
        need_insulin          = st.checkbox("Insulin data",          value=False)
        need_carbs            = st.checkbox("Carbs data",            value=False)
        need_heartrate        = st.checkbox("Heart Rate data",       value=False)
        need_steps            = st.checkbox("Steps data",            value=False)
        need_workout_duration = st.checkbox("Workout Duration",      value=False)
        need_workout_intensity= st.checkbox("Workout Intensity",     value=False)
        need_workout_label    = st.checkbox("Workout Label",         value=False)

        st.markdown("**Age range:**")
        col_age1, col_age2 = st.columns(2)
        with col_age1:
            min_age_input = st.number_input("Min age", min_value=0, max_value=120, value=0, step=1)
        with col_age2:
            max_age_input = st.number_input("Max age", min_value=0, max_value=120, value=120, step=1)
        min_age = min_age_input if min_age_input > 0 else None
        max_age = max_age_input if max_age_input < 120 else None

        st.markdown("**Therapy type:**")
        all_therapies = get_therapy_types()
        therapy_filter = st.multiselect("Treatment group", all_therapies, placeholder="All therapies")

        col_ps, col_all_check = st.columns([3, 1])
        with col_ps:
            page_size = st.number_input("Patients per page", min_value=1, max_value=500, value=10, step=5)
        with col_all_check:
            st.markdown("<br>", unsafe_allow_html=True)
            load_all_check = st.checkbox("All", value=False)
        if load_all_check:
            page_size = 999999

        filter_key = (f"{need_cgm}_{need_insulin}_{need_carbs}_{need_heartrate}_{need_steps}_"
                      f"{need_workout_duration}_{need_workout_intensity}_{need_workout_label}_"
                      f"{min_age}_{max_age}_{str(therapy_filter)}_{page_size}")
        if "filter_key" not in st.session_state or st.session_state["filter_key"] != filter_key:
            st.session_state["filter_key"]  = filter_key
            st.session_state["filter_page"] = 1
            st.session_state["search_all"]  = False

        page       = st.session_state.get("filter_page", 1)
        search_all = st.session_state.get("search_all", False)
        page_end   = page * int(page_size)

        _filter_kwargs = dict(
            need_cgm=need_cgm, need_insulin=need_insulin, need_carbs=need_carbs,
            need_heartrate=need_heartrate, need_steps=need_steps,
            need_workout_duration=need_workout_duration,
            need_workout_intensity=need_workout_intensity,
            need_workout_label=need_workout_label,
            therapy_filter=therapy_filter,
            min_age=min_age, max_age=max_age,
        )

        if search_all:
            filtered  = get_filtered_patients(**_filter_kwargs, limit=None)
            shown     = filtered
            has_more  = False
            label     = f"<b>{len(filtered)}</b> patients found"
        else:
            filtered = get_filtered_patients(**_filter_kwargs, limit=page_end)
            shown    = filtered
            has_more = len(get_filtered_patients(**_filter_kwargs, limit=page_end + 1)) > page_end
            if has_more:
                label = f"Showing first <b>{len(shown)}</b> — more available"
            else:
                label = f"<b>{len(shown)}</b> patients found"

        if len(shown) > 0:
            st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            selected = st.selectbox("Patient ID", ["— select —"] + shown)
            col_more, col_all = st.columns(2)
            with col_more:
                if has_more and st.button(f"Load {int(page_size)} more", use_container_width=True):
                    st.session_state["filter_page"] = page + 1
                    st.rerun()
            with col_all:
                if not search_all and st.button("Search all", use_container_width=True):
                    st.session_state["search_all"] = True
                    st.rerun()
        else:
            st.warning("No patients match the selected filters.")
            selected = "— select —"

    st.divider()
    st.markdown("### Options")
    window_days = st.slider("Days per window", 1, 7, 2)

    if selected != "— select —":
        p_min_temp, _ = get_patient_dates(selected)
        off_temp = st.session_state.get("offset", 0)
        window_start = p_min_temp.normalize() + timedelta(days=off_temp)
        window_end = window_start + timedelta(days=window_days)

    if selected != "— select —" and 'filtered' in locals() and len(shown) > 0:
            st.divider()
            st.markdown("### 💾 MATLAB Export")
            st.caption(f"Genera .mat per TUTTI i {len(shown)} pazienti trovati")
    
    if st.button("Preparazione file .mat", use_container_width=True):
        mat_data = {}
        progress_bar = st.progress(0)
        
        for i, p_id in enumerate(shown):
            p_min_full, p_max_full = get_patient_dates(p_id)
            p_df = get_window_data(p_id, p_min_full, p_max_full)
            
            if not p_df.empty:
                safe_id = f"id_{p_id.replace('-', '_')}"
                mat_data[safe_id] = {
                    'patient_id': p_id,
                    'time': p_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values,
                    'cgm': p_df['CGM'].values,
                    'carbs': p_df['carbs'].values,
                    'bolus': p_df['bolus'].values,
                    'insulin': p_df['insulin'].values,
                    'basal': p_df['basal'].values,
                    'heartrate': p_df['heartrate'].values if 'heartrate' in p_df else [],
                    'steps': p_df['steps'].values if 'steps' in p_df else [],
                    'workout_duration': p_df['workout_duration'].values if 'workout_duration' in p_df else [],
                    'workout_intensity': p_df['workout_intensity'].values if 'workout_intensity' in p_df else [],
                }
            
            progress_bar.progress((i + 1) / len(shown))
        
        if mat_data:
            buf = io.BytesIO()
            savemat(buf, mat_data)
            progress_bar.empty()
            
            st.download_button(
                label="📥 Scarica .mat",
                data=buf.getvalue(),
                file_name=f"metabonet_page_data.mat",
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            st.error("Nessun dato trovato per questi pazienti.")

# ── Stop if no patient selected ───────────────────────────────────────────────
if selected in ["— select —", "— seleziona —"]:
    st.info("Select a patient from the sidebar to visualize data.")
    st.stop()

# ── Reset offset on patient change ───────────────────────────────────────────
if "last_patient" not in st.session_state or st.session_state["last_patient"] != selected:
    st.session_state["offset"] = 0
    st.session_state["last_patient"] = selected

# ── Patient date range ────────────────────────────────────────────────────────
pat_min, pat_max = get_patient_dates(selected)
pat_min    = pat_min.normalize()
pat_max    = pat_max.normalize()
total_days = (pat_max - pat_min).days
max_offset = max(0, total_days - window_days + 1)

# ── Patient timeline overview ─────────────────────────────────────────────────
st.markdown("#### Patient Timeline — click a day to navigate")

daily = get_daily_coverage(
    selected, need_cgm, need_insulin, need_carbs,
    need_heartrate=need_heartrate,
    need_steps=need_steps,
    need_workout_duration=need_workout_duration,
    need_workout_intensity=need_workout_intensity,
    need_workout_label=need_workout_label,
)
offset = st.session_state.get("offset", 0)
current_day = pat_min + timedelta(days=offset)

timeline_fig = go.Figure()

for _, row in daily.iterrows():
    color     = "#2ecc71" if row['complete'] else "#dde3ea"
    border    = "#27ae60" if row['complete'] else "#b0b8c1"
    day_ts    = row['day']
    day_center = day_ts + timedelta(hours=12)
    is_current = (day_ts >= current_day) and (day_ts < current_day + timedelta(days=window_days))
    border_w  = 2.5 if is_current else 0.5

    timeline_fig.add_trace(go.Bar(
        x=[day_center],
        y=[1],
        width=86400000,
        marker=dict(color=color, line=dict(color=border, width=border_w)),
        hovertemplate=(
            f"<b>{day_ts.strftime('%d %b %Y')}</b><br>"
            f"CGM: {'✓' if row['has_cgm'] else '✗'}<br>"
            f"Carbs: {'✓' if row['has_carbs'] else '✗'}<br>"
            f"Insulin: {'✓' if row['has_insulin'] else '✗'}<br>"
            f"Heart Rate: {'✓' if row['has_heartrate'] else '✗'}<br>"
            f"Steps: {'✓' if row['has_steps'] else '✗'}<br>"
            f"Workout Duration: {'✓' if row['has_workout_duration'] else '✗'}<br>"
            f"Workout Intensity: {'✓' if row['has_workout_intensity'] else '✗'}<br>"
            f"Workout Label: {'✓' if row['has_workout_label'] else '✗'}<br>"
            "<i>Click to navigate</i><extra></extra>"
        ),
        showlegend=False,
    ))

timeline_fig.update_layout(
    height=80,
    margin=dict(l=10, r=10, t=5, b=5),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, range=[pat_min - timedelta(days=1), pat_max + timedelta(days=1)],
               tickformat="%b %Y", tickangle=0),
    yaxis=dict(visible=False, range=[0, 1.5]),
    barmode='overlay',
    clickmode='event+select',
)

timeline_fig.add_vrect(
    x0=current_day, x1=current_day + timedelta(days=window_days),
    fillcolor="rgba(41,128,185,0.15)", line_color="#2980b9", line_width=1.5,
)

timeline_event = st.plotly_chart(timeline_fig, use_container_width=True, on_select="rerun", key="timeline")

if timeline_event and timeline_event.get("selection") and timeline_event["selection"].get("points"):
    clicked_x = timeline_event["selection"]["points"][0].get("x")
    if clicked_x:
        clicked_day = pd.Timestamp(clicked_x).normalize()
        new_offset  = (clicked_day - pat_min).days
        new_offset  = max(0, min(new_offset, max_offset))
        if new_offset != st.session_state["offset"]:
            st.session_state["offset"] = new_offset
            st.rerun()

offset = st.session_state.get("offset", 0)
window_start = pat_min + timedelta(days=offset)
window_end   = window_start + timedelta(days=window_days)


st.divider()


# ── patient characteristics ───────────────────────────────────────────────────
meta = get_patient_metadata(selected)

st.markdown("### Patient Profile")

col_id, col_m1, col_m2, col_m3, col_m4 = st.columns([1.5, 1, 1, 1, 1.5])

with col_id:
    st.markdown(f"**Patient ID:** {meta['id']}")

with col_m1:
    age_val = f"{int(meta['age'])}" if pd.notnull(meta['age']) else "N/A"
    st.markdown(f"**Age:** {age_val}")
    st.markdown(f"**Gender:** {meta['gender'] if pd.notnull(meta['gender']) else 'N/A'}")

with col_m2:
    st.markdown(f"**Ethnicity:** {meta['ethnicity'] if pd.notnull(meta['ethnicity']) else 'N/A'}")
    diag_val = f"{int(meta['age_diagnosis'])}" if pd.notnull(meta['age_diagnosis']) else "N/A"
    st.markdown(f"**Diagnosis Age:** {diag_val}")

with col_m3:
    weight_val = f"{meta['weight']:.1f} kg" if pd.notnull(meta['weight']) else "N/A"
    st.markdown(f"**Weight:** {weight_val}")
    height_val = f"{meta['height']:.1f} cm" if pd.notnull(meta['height']) else "N/A"
    st.markdown(f"**Height:** {height_val}")

with col_m4:
    st.markdown(f"**Therapy:** `{meta['treatment_group'] if pd.notnull(meta['treatment_group']) else 'N/A'}`")
    st.markdown(f"**Device:** {meta['cgm_device'] if pd.notnull(meta['cgm_device']) else 'N/A'}")

st.divider()

# ── Navigation ────────────────────────────────────────────────────────────────
col_prev, col_label, col_next = st.columns([1, 4, 1])
with col_prev:
    if st.button("Previous", use_container_width=True, disabled=(offset <= 0)):
        st.session_state["offset"] = max(0, offset - window_days)
        st.rerun()
with col_label:
    end_label = min(window_end, pat_max)
    st.markdown(
        f'<div class="nav-label">{window_start.strftime("%d %b %Y")} — {end_label.strftime("%d %b %Y")}</div>',
        unsafe_allow_html=True,
    )
with col_next:
    if st.button("Next", use_container_width=True, disabled=(offset >= max_offset)):
        st.session_state["offset"] = min(max_offset, offset + window_days)
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Load window data ──────────────────────────────────────────────────────────
df = get_window_data(selected, window_start, window_end)
df['date'] = pd.to_datetime(df['date'])

df_cgm   = df[df['CGM'].notna()].copy()
df_meals = df[df['carbs'].notna() & (df['carbs'] > 0)].copy()
df_bolus = df[df['bolus'].notna()  & (df['bolus']  > 0)].copy()
df_ins   = df[df['insulin'].notna() & (df['insulin'] > 0)].copy()
df_basal = df[df['basal'].notna()  & (df['basal']  > 0)].copy()

# ── New signal dataframes ─────────────────────────────────────────────────────
df_hr       = df[df['heartrate'].notna() & (df['heartrate'] > 0)].copy()     if 'heartrate'         in df.columns else pd.DataFrame()
df_steps    = df[df['steps'].notna()     & (df['steps']     > 0)].copy()     if 'steps'             in df.columns else pd.DataFrame()
df_wdur     = df[df['workout_duration'].notna() & (df['workout_duration'] > 0)].copy() if 'workout_duration'  in df.columns else pd.DataFrame()
df_wint     = df[df['workout_intensity'].notna() & (df['workout_intensity'] > 0)].copy() if 'workout_intensity' in df.columns else pd.DataFrame()
df_wlab     = df.copy() if 'workout_label' in df.columns else pd.DataFrame()
if not df_wlab.empty:
    try:
        df_wlab = df_wlab[pd.to_numeric(df_wlab['workout_label'], errors='coerce').fillna(0) > 0]
    except Exception:
        df_wlab = pd.DataFrame()

has_cgm   = not df_cgm.empty
has_meals = not df_meals.empty
has_ins   = not df_bolus.empty or not df_ins.empty or not df_basal.empty
has_activity = not df_hr.empty or not df_steps.empty or not df_wdur.empty or not df_wint.empty or not df_wlab.empty

# ── Debug expander ────────────────────────────────────────────────────────────
with st.expander("Debug — current window data", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total rows", len(df))
    d2.metric("Valid CGM rows", len(df_cgm))
    d3.metric("Carbs > 0", len(df_meals))
    d4.metric("Insulin > 0", len(df_bolus) + len(df_ins))
    null_info = pd.DataFrame({
        "Column": ["CGM","carbs","bolus","insulin","basal","heartrate","steps","workout_duration","workout_intensity"],
        "Valid values": [
            int(df['CGM'].notna().sum()),
            int((df['carbs']>0).sum()) if df['carbs'].notna().any() else 0,
            int((df['bolus']>0).sum()) if df['bolus'].notna().any() else 0,
            int((df['insulin']>0).sum()) if df['insulin'].notna().any() else 0,
            int((df['basal']>0).sum()) if df['basal'].notna().any() else 0,
            int((df['heartrate']>0).sum()) if 'heartrate' in df.columns and df['heartrate'].notna().any() else 0,
            int((df['steps']>0).sum()) if 'steps' in df.columns and df['steps'].notna().any() else 0,
            int((df['workout_duration']>0).sum()) if 'workout_duration' in df.columns and df['workout_duration'].notna().any() else 0,
            int((df['workout_intensity']>0).sum()) if 'workout_intensity' in df.columns and df['workout_intensity'].notna().any() else 0,
        ],
        "NULL": [df[c].isna().sum() if c in df.columns else "-" for c in ['CGM','carbs','bolus','insulin','basal','heartrate','steps','workout_duration','workout_intensity']],
    })
    st.dataframe(null_info, use_container_width=True, hide_index=True)

# ── No data warning ───────────────────────────────────────────────────────────
if not has_cgm and not has_meals and not has_ins and not has_activity:
    st.warning("No data available for this time window. Use Next or click on the timeline.")
    st.stop()

# ── CGM metrics ───────────────────────────────────────────────────────────────
if has_cgm:
    tir      = ((df_cgm['CGM'] >= 70) & (df_cgm['CGM'] <= 180)).mean() * 100
    hypo_70  = (df_cgm['CGM'] < 70).mean()  * 100
    hypo_80  = (df_cgm['CGM'] < 80).mean()  * 100
    hypo_50  = (df_cgm['CGM'] < 50).mean()  * 100
    hyper_180= (df_cgm['CGM'] > 180).mean() * 100
    hyper_140= (df_cgm['CGM'] > 140).mean() * 100
    avg_cgm  = df_cgm['CGM'].mean()
    sd_cgm   = df_cgm['CGM'].std()
    cv_cgm   = (sd_cgm / avg_cgm) * 100 if avg_cgm > 0 else 0
    tr       = 100 - (hypo_70  + hyper_180)
    ttr      = 100 - (hypo_80  + hyper_140)

    # Card CSS più compatte
    st.markdown("""
    <style>
    .metric-card-sm {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 10px; padding: 0.55rem 0.4rem;
        text-align: center;
    }
    .metric-val-sm { font-size: 1.25rem; font-weight: 700; }
    .metric-lbl-sm { font-size: 0.65rem; color: #888;
                     text-transform: uppercase; letter-spacing: 0.08em; }
    </style>
    """, unsafe_allow_html=True)

    def metric_card(col, val, lbl, color):
            col.markdown(
                f'<div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; '
                f'padding:0.4rem 0.6rem; text-align:center; white-space:nowrap;">'
                f'<span style="font-size:0.65rem; color:#888; text-transform:uppercase; '
                f'letter-spacing:0.26em;">{lbl}: </span>'
                f'<span style="font-size:1.05rem; font-weight:700; color:{color};">{val}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Riga 1 — statistiche di base
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{avg_cgm:.1f} mg/dL", "Average CGM",   "#3498db")
    metric_card(c2, f"{sd_cgm:.1f} mg/dL",  "SD CGM",        "#3498db")
    metric_card(c3, f"{cv_cgm:.1f}%",        "CV CGM",        "#3498db")
    metric_card(c4, f"{ttr:.1f}%",           "TIR tight 80–140", "#2ecc71")

    st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)

    # Riga 2 — range e soglie
    c5, c6, c7, c8 = st.columns(4)
    metric_card(c5, f"{tr:.1f}%",       "TIR 70–180",  "#46d446")
    metric_card(c6, f"{hyper_180:.1f}%","Time > 180",  "#f39c12")
    metric_card(c7, f"{hypo_70:.1f}%",  "Time < 70",   "#f39c12")
    metric_card(c8, f"{hypo_50:.1f}%",  "Time < 50",   "#e74c3c")

    st.markdown("<br>", unsafe_allow_html=True)

# ── Dynamic subplots ──────────────────────────────────────────────────────────
rows = []
if has_cgm or has_meals:  rows.append(("CGM (mg/dL) & Carbs (g)", 0.50))
if has_ins:               rows.append(("Insulin / Basal",         0.25))
if has_activity:          rows.append(("Activity",                0.25))

n_rows      = len(rows)
row_heights = [r[1]/sum(r[1] for r in rows) for r in rows]

# secondary_y only on first subplot (CGM + carbs)
specs = []
for i in range(n_rows):
    specs.append([{"secondary_y": (i == 0)}])

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.06,
    subplot_titles=[r[0] for r in rows],
    specs=specs,
)

# Map subplot labels to row indices
cgm_row      = next((i+1 for i, r in enumerate(rows) if "CGM" in r[0]),      None)
ins_row      = next((i+1 for i, r in enumerate(rows) if "Insulin" in r[0]),  None)
activity_row = next((i+1 for i, r in enumerate(rows) if "Activity" in r[0]), None)

x_range = [window_start, window_end]

# ── CGM + meals ───────────────────────────────────────────────────────────────
if cgm_row:


    if has_cgm:
        fig.add_trace(go.Scatter(
            x=df_cgm['date'], y=df_cgm['CGM'],
            mode='lines', name='CGM',
            line=dict(color='#2980b9', width=1.8),
        ), row=cgm_row, col=1, secondary_y=False)

    if has_meals:
        fig.add_trace(go.Scatter(
            x=df_meals['date'], y=df_meals['carbs'],
            mode='markers', name='Meal (g)',
            marker=dict(symbol='circle', size=11, color='#2ecc71',
                        line=dict(color='white', width=1.5)),
            hovertemplate='<b>Meal</b><br>Carbs: %{y} g<br>%{x|%H:%M}<extra></extra>',
        ), row=cgm_row, col=1, secondary_y=True)    
        
    
    # Rosso: sotto 50 (ipoglicemia severa)
    fig.add_hrect(y0=0,   y1=50,  fillcolor="rgba(231,76,60,0.10)",   line_width=0, row=cgm_row, col=1)
    # Verdino: tra 50 e 70 (ipoglicemia lieve)
    fig.add_hrect(y0=50,  y1=70,  fillcolor="rgba(211,84,0,0.05)",   line_width=0, row=cgm_row, col=1)
    # Arancione scuro: tra 70 e 80 (zona limite bassa)
    fig.add_hrect(y0=70,  y1=80,  fillcolor="rgba(39,174,96,0.03)",    line_width=0, row=cgm_row, col=1)# rgba(211,84,0,0.20)
    # Verde: range ottimale 80–140
    fig.add_hrect(y0=80,  y1=140, fillcolor="rgba(46,204,113,0.15)",  line_width=0, row=cgm_row, col=1)
    # Verdino: tra 140 e 180 (zona limite alta)
    fig.add_hrect(y0=140, y1=180, fillcolor="rgba(39,174,96,0.05)",   line_width=0, row=cgm_row, col=1)
    # Arancione scuro: sopra 180 (iperglicemia)
    fig.add_hrect(y0=180, y1=400, fillcolor="rgba(211,84,0,0.05)",    line_width=0, row=cgm_row, col=1)
        


# ── Insulin ───────────────────────────────────────────────────────────────────
if ins_row:
    if not df_basal.empty:
        fig.add_trace(go.Scatter(
            x=df_basal['date'], y=df_basal['basal'],
            mode='lines', name='Basal (U/h)',
            line=dict(color='#16a085', width=1.5),
            fill='tozeroy', fillcolor='rgba(22,160,133,0.15)',
        ), row=ins_row, col=1)
    if not df_bolus.empty:
        fig.add_trace(go.Bar(x=df_bolus['date'], y=df_bolus['bolus'],
                             name='Bolus (U)', marker_color='#8e44ad'), row=ins_row, col=1)
    if not df_ins.empty:
        fig.add_trace(go.Bar(x=df_ins['date'], y=df_ins['insulin'],
                             name='Insulin (U)', marker_color='#2c3e50'), row=ins_row, col=1)

# ── Activity subplot ──────────────────────────────────────────────────────────
if activity_row:
    # Heart Rate — line
    if not df_hr.empty:
        fig.add_trace(go.Scatter(
            x=df_hr['date'], y=df_hr['heartrate'],
            mode='lines', name='Heart Rate (bpm)',
            line=dict(color='#e74c3c', width=1.5),
        ), row=activity_row, col=1)

    # Steps — filled area
    if not df_steps.empty:
        fig.add_trace(go.Scatter(
            x=df_steps['date'], y=df_steps['steps'],
            mode='lines', name='Steps',
            line=dict(color='#9b59b6', width=1.2),
            fill='tozeroy', fillcolor='rgba(155,89,182,0.15)',
        ), row=activity_row, col=1)

    # Workout Duration — bars
    if not df_wdur.empty:
        fig.add_trace(go.Bar(
            x=df_wdur['date'], y=df_wdur['workout_duration'],
            name='Workout Duration (min)',
            marker_color='rgba(243,156,18,0.7)',
            marker_line_color='#e67e22', marker_line_width=1,
        ), row=activity_row, col=1)

    # Workout Intensity — markers
    if not df_wint.empty:
        fig.add_trace(go.Scatter(
            x=df_wint['date'], y=df_wint['workout_intensity'],
            mode='markers', name='Workout Intensity',
            marker=dict(symbol='diamond', size=9, color='#e67e22',
                        line=dict(color='white', width=1)),
            hovertemplate='<b>Intensity</b>: %{y}<br>%{x|%H:%M}<extra></extra>',
        ), row=activity_row, col=1)

    # Workout Label — scatter with label text
    if not df_wlab.empty:
        fig.add_trace(go.Scatter(
            x=df_wlab['date'],
            y=[0] * len(df_wlab),
            mode='markers+text',
            name='Workout Label',
            text=df_wlab['workout_label'].astype(str),
            textposition='top center',
            marker=dict(symbol='triangle-up', size=10, color='#c0392b'),
            hovertemplate='<b>Label</b>: %{text}<br>%{x|%H:%M}<extra></extra>',
        ), row=activity_row, col=1)

# ── Axis configuration ────────────────────────────────────────────────────────
fig.update_xaxes(range=x_range, showgrid=True, gridcolor="#f0f0f0", tickformat="%d %b\n%H:%M")

if cgm_row:
    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[0, 400],
                     showgrid=True, gridcolor="#eeeeee", row=cgm_row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Carbs (g)", range=[0, 250],
                     showgrid=False, row=cgm_row, col=1, secondary_y=True)

if ins_row:
    fig.update_yaxes(title_text="U / U/h", showgrid=True, gridcolor="#eeeeee", row=ins_row, col=1)

if activity_row:
    fig.update_yaxes(title_text="Activity", showgrid=True, gridcolor="#eeeeee", row=activity_row, col=1)

fig.update_layout(
    template="plotly_white",
    height=300 + 220 * n_rows,   # scale height with number of subplots
    margin=dict(l=10, r=10, t=50, b=10),
    hovermode="x",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#dee2e6", borderwidth=1
    ),
)

st.plotly_chart(fig, use_container_width=True)