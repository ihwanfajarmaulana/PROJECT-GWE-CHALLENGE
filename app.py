
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    APP_DIR / "ai_student_impact_dataset (1).csv",
    APP_DIR.parent / ".cache" / "ai_student_impact_dataset (1).csv",
]
MODEL_CANDIDATES = [
    APP_DIR / "burnout_model.pkl",
]
ENCODER_PATH = APP_DIR / "label_encoder.pkl"
METRICS_PATH = APP_DIR / "model_evaluation.csv"

COLORS = {
    "navy": "#102A43",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "green": "#10B981",
    "yellow": "#F59E0B",
    "red": "#EF4444",
    "muted": "#64748B",
    "surface": "#FFFFFF",
    "background": "#F4F7FB",
}
RISK_COLORS = {
    "Low": COLORS["green"],
    "Medium": COLORS["yellow"],
    "High": COLORS["red"],
}

st.set_page_config(
    page_title="Student Burnout Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background: #F4F7FB;
            color: #102A43;
        }

        [data-testid="stSidebar"] {
            background: #102A43;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 0.55rem 0.7rem;
            border-radius: 6px;
            margin-bottom: 0.25rem;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.12);
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #102A43;
            letter-spacing: 0;
        }

        .page-eyebrow {
            color: #2563EB;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .page-title {
            color: #102A43;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }

        .page-subtitle {
            color: #64748B;
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 850px;
            margin-top: 0.55rem;
            margin-bottom: 1.5rem;
        }

        .hero {
            min-height: 310px;
            padding: 2.6rem;
            border-radius: 8px;
            background:
                linear-gradient(110deg, rgba(9, 30, 66, 0.97), rgba(37, 99, 235, 0.88)),
                radial-gradient(circle at 80% 20%, #22D3EE 0, transparent 33%);
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 16px 40px rgba(15, 42, 67, 0.16);
        }

        .hero-tag {
            display: inline-block;
            width: fit-content;
            padding: 0.35rem 0.7rem;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.08);
        }

        .hero h1 {
            color: white;
            max-width: 760px;
            font-size: 2.65rem;
            line-height: 1.12;
            margin: 0;
        }

        .hero p {
            max-width: 720px;
            color: #DCEBFF;
            line-height: 1.7;
            margin: 1rem 0 0;
        }

        .metric-card, .info-card, .result-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 6px 18px rgba(15, 42, 67, 0.05);
        }

        .metric-label {
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            color: #102A43;
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        .metric-note {
            color: #64748B;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .section-title {
            color: #102A43;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1.7rem 0 0.7rem;
        }

        .info-card h3 {
            font-size: 1rem;
            margin: 0 0 0.45rem;
        }

        .info-card p {
            color: #64748B;
            font-size: 0.88rem;
            line-height: 1.6;
            margin: 0;
        }

        .insight-box {
            padding: 1rem 1.1rem;
            border-left: 4px solid #2563EB;
            border-radius: 0 6px 6px 0;
            background: #EFF6FF;
            color: #334E68;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .status-chip {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #DCFCE7;
            color: #166534;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .footer {
            color: #94A3B8;
            text-align: center;
            font-size: 0.78rem;
            padding-top: 2rem;
        }

        div[data-testid="stForm"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.2rem;
        }

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] label p,
        div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
            color: #334E68 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }

        div[data-testid="stForm"] small {
            color: #64748B !important;
            opacity: 1 !important;
        }

        div[data-testid="stForm"] [data-baseweb="select"] *,
        div[data-testid="stForm"] input {
            color: #102A43 !important;
        }

        div[data-testid="stForm"] [data-testid="stSlider"] p {
            color: #334E68 !important;
            opacity: 1 !important;
        }

        div[data-testid="stForm"] [data-testid="stSliderThumbValue"] {
            color: #EF4444 !important;
        }

        /* Pastikan semua label widget terbaca pada tema browser terang/gelap. */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stSlider"] p {
            color: #334E68 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        /* Input dan filter menggunakan permukaan terang dengan teks gelap. */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stNumberInput"] > div > div {
            background: #FFFFFF !important;
            border-color: #CBD5E1 !important;
            color: #102A43 !important;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        [data-testid="stNumberInput"] input {
            color: #102A43 !important;
        }

        div[data-baseweb="tag"] {
            background: #DBEAFE !important;
        }

        div[data-baseweb="tag"] span,
        div[data-baseweb="tag"] svg {
            color: #1D4ED8 !important;
            fill: #1D4ED8 !important;
        }

        /* Expander filter dan data pendukung. */
        [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #D7E2F0 !important;
            border-radius: 8px !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            background: #EAF2FF !important;
        }

        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary svg {
            color: #173B67 !important;
            fill: #173B67 !important;
            font-weight: 700 !important;
        }

        /* Tab dokumentasi tetap terbaca meskipun browser memakai dark mode. */
        [data-testid="stTabs"] button {
            color: #52677D !important;
        }

        [data-testid="stTabs"] button p {
            color: #52677D !important;
            font-weight: 600 !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] p {
            color: #1D4ED8 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #2563EB !important;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 44px;
            background: #2563EB;
            color: white;
            border: 0;
            border-radius: 6px;
            font-weight: 700;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #1D4ED8;
            color: white;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero {
                min-height: auto;
                padding: 1.5rem;
            }
            .hero h1 {
                font-size: 1.8rem;
            }
            .page-title {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data():
    for data_path in DATA_CANDIDATES:
        if data_path.exists():
            return pd.read_csv(data_path), data_path
    return None, None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None or not ENCODER_PATH.exists():
        return None, None
    return joblib.load(model_path), joblib.load(ENCODER_PATH)


@st.cache_data(show_spinner=False)
def load_evaluation():
    if not METRICS_PATH.exists():
        return None
    return pd.read_csv(METRICS_PATH)


def get_model_name(model):
    if model is None:
        return "Belum tersedia"
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model = model.named_steps["model"]
    readable_names = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
    }
    class_name = model.__class__.__name__
    return readable_names.get(class_name, class_name)


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def style_figure(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=55, b=16),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color=COLORS["navy"]),
        title_font=dict(size=16, color=COLORS["navy"]),
        legend_title_text="",
        legend=dict(font=dict(color=COLORS["navy"])),
        hoverlabel=dict(bgcolor="#102A43", font_color="white"),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        tickfont=dict(color=COLORS["navy"]),
        title_font=dict(color=COLORS["navy"]),
    )
    fig.update_yaxes(
        gridcolor="#EDF2F7",
        zeroline=False,
        tickfont=dict(color=COLORS["navy"]),
        title_font=dict(color=COLORS["navy"]),
    )
    return fig


def format_number(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 1rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: white;">
                    Student Burnout
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.25rem;">
                    GenAI Impact Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "NAVIGASI",
            [
                "Beranda",
                "Dashboard EDA",
                "Prediksi Burnout",
                "Dokumentasi",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.6;">
                Metode: CRISP-DM<br>
                Subtema: Education & Risk Prediction<br>
                Versi aplikasi: 1.0
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


def render_home(df):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">Education Analytics • Early Warning System</div>
            <h1>Prediksi Risiko Burnout Mahasiswa di Era Generative AI</h1>
            <p>
                Dashboard ini menganalisis hubungan penggunaan GenAI, pola belajar,
                kecemasan ujian, dan ketergantungan AI dengan tingkat risiko burnout
                mahasiswa. Model klasifikasi memberikan prediksi Low, Medium, atau High
                sebagai dukungan untuk tindakan pencegahan lebih awal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Ringkasan proyek")
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Jumlah Data", f"{len(df):,}".replace(",", "."), "Record mahasiswa")
        with c2:
            metric_card("Jumlah Variabel", str(df.shape[1]), "Fitur dan target")
        with c3:
            metric_card(
                "Rata-rata GenAI",
                f"{format_number(df['Weekly_GenAI_Hours'].mean())} jam",
                "Penggunaan per minggu",
            )
        with c4:
            high_rate = (df["Burnout_Risk_Level"] == "High").mean() * 100
            metric_card("Risiko Tinggi", f"{format_number(high_rate)}%", "Dari seluruh mahasiswa")
    else:
        st.warning("Dataset belum ditemukan. Letakkan file CSV di folder yang sama dengan app.py.")

    section_title("Latar belakang dan tujuan")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
            <div class="info-card">
                <h3>Latar Belakang Masalah</h3>
                <p>
                    Generative AI membantu mahasiswa menyusun ide, meringkas bacaan,
                    melakukan debugging, dan menyelesaikan tugas. Namun, penggunaan
                    yang tidak terkontrol dapat berkaitan dengan ketergantungan AI,
                    berkurangnya pola belajar tradisional, kecemasan, dan burnout.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3>Tujuan Proyek</h3>
                <p>
                    Menghasilkan eksplorasi data yang informatif, mengetahui faktor
                    yang berkaitan dengan burnout, membangun model klasifikasi, dan
                    menyediakan prediksi risiko secara real-time melalui Streamlit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Fitur aplikasi")
    c1, c2, c3 = st.columns(3)
    cards = [
        (
            c1,
            "Dashboard EDA",
            "Filter data dan visualisasi interaktif untuk mengeksplorasi karakteristik mahasiswa.",
        ),
        (
            c2,
            "Prediksi Real-Time",
            "Masukkan kondisi mahasiswa dan dapatkan tingkat risiko beserta probabilitasnya.",
        ),
        (
            c3,
            "Dokumentasi Model",
            "Informasi metode CRISP-DM, fitur, evaluasi, penggunaan aplikasi, dan tim.",
        ),
    ]
    for column, title, description in cards:
        with column:
            st.markdown(
                f'<div class="info-card"><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )


def render_eda(df):
    page_header(
        "Exploratory Data Analysis",
        "Dashboard Eksplorasi Data",
        "Gunakan filter untuk melihat pola penggunaan GenAI, kondisi belajar, dan risiko burnout pada kelompok mahasiswa tertentu.",
    )

    if df is None:
        st.error("Dataset belum ditemukan. Pastikan file CSV berada di folder aplikasi.")
        return

    with st.expander("Filter data", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            selected_major = st.multiselect(
                "Kategori jurusan",
                sorted(df["Major_Category"].dropna().unique()),
                default=sorted(df["Major_Category"].dropna().unique()),
            )
        with f2:
            selected_year = st.multiselect(
                "Tahun studi",
                sorted(df["Year_of_Study"].dropna().unique()),
                default=sorted(df["Year_of_Study"].dropna().unique()),
            )
        with f3:
            selected_risk = st.multiselect(
                "Risiko burnout",
                ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"],
            )
        with f4:
            gpa_range = st.slider(
                "Rentang GPA awal",
                min_value=float(df["Pre_Semester_GPA"].min()),
                max_value=float(df["Pre_Semester_GPA"].max()),
                value=(
                    float(df["Pre_Semester_GPA"].min()),
                    float(df["Pre_Semester_GPA"].max()),
                ),
                step=0.01,
            )

    filtered = df[
        df["Major_Category"].isin(selected_major)
        & df["Year_of_Study"].isin(selected_year)
        & df["Burnout_Risk_Level"].isin(selected_risk)
        & df["Pre_Semester_GPA"].between(gpa_range[0], gpa_range[1])
    ].copy()

    if filtered.empty:
        st.warning("Tidak ada data yang sesuai dengan kombinasi filter tersebut.")
        return

    c1, c2, c3, c4 = st.columns(4)
    high_rate = (filtered["Burnout_Risk_Level"] == "High").mean() * 100
    with c1:
        metric_card("Data Terpilih", f"{len(filtered):,}".replace(",", "."), "Setelah filter")
    with c2:
        metric_card(
            "GenAI Mingguan",
            f"{format_number(filtered['Weekly_GenAI_Hours'].mean())} jam",
            "Nilai rata-rata",
        )
    with c3:
        metric_card(
            "Ketergantungan AI",
            format_number(filtered["Perceived_AI_Dependency"].mean()),
            "Skala 1 sampai 10",
        )
    with c4:
        metric_card("Risiko Tinggi", f"{format_number(high_rate)}%", "Proporsi data terpilih")

    section_title("Komposisi risiko dan pola penggunaan")
    left, right = st.columns(2)
    with left:
        risk_counts = (
            filtered["Burnout_Risk_Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )
        risk_counts.columns = ["Burnout_Risk_Level", "Jumlah"]
        fig = px.pie(
            risk_counts,
            names="Burnout_Risk_Level",
            values="Jumlah",
            hole=0.62,
            title="Distribusi Risiko Burnout",
            color="Burnout_Risk_Level",
            color_discrete_map=RISK_COLORS,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(style_figure(fig), use_container_width=True)

    with right:
        major_risk = (
            filtered.groupby(["Major_Category", "Burnout_Risk_Level"], observed=True)
            .size()
            .reset_index(name="Jumlah")
        )
        fig = px.bar(
            major_risk,
            x="Major_Category",
            y="Jumlah",
            color="Burnout_Risk_Level",
            barmode="group",
            title="Risiko Burnout Berdasarkan Jurusan",
            category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]},
            color_discrete_map=RISK_COLORS,
        )
        fig.update_xaxes(title="Kategori jurusan")
        fig.update_yaxes(title="Jumlah mahasiswa")
        st.plotly_chart(style_figure(fig), use_container_width=True)

    section_title("Hubungan faktor utama dengan burnout")
    chart_variable = st.selectbox(
        "Pilih variabel yang dianalisis",
        [
            "Weekly_GenAI_Hours",
            "Traditional_Study_Hours",
            "Perceived_AI_Dependency",
            "Anxiety_Level_During_Exams",
            "Pre_Semester_GPA",
        ],
        format_func={
            "Weekly_GenAI_Hours": "Penggunaan GenAI per Minggu",
            "Traditional_Study_Hours": "Jam Belajar Tradisional",
            "Perceived_AI_Dependency": "Ketergantungan AI",
            "Anxiety_Level_During_Exams": "Kecemasan Saat Ujian",
            "Pre_Semester_GPA": "GPA Sebelum Semester",
        }.get,
    )

    left, right = st.columns([1.25, 1])
    with left:
        fig = px.box(
            filtered,
            x="Burnout_Risk_Level",
            y=chart_variable,
            color="Burnout_Risk_Level",
            title=f"Distribusi {chart_variable} pada Setiap Tingkat Risiko",
            category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]},
            color_discrete_map=RISK_COLORS,
            points=False,
        )
        fig.update_xaxes(title="Tingkat risiko burnout")
        fig.update_yaxes(title=chart_variable)
        st.plotly_chart(style_figure(fig, 430), use_container_width=True)

    with right:
        summary = (
            filtered.groupby("Burnout_Risk_Level", observed=True)[chart_variable]
            .agg(["mean", "median", "std"])
            .reindex(["Low", "Medium", "High"])
            .round(2)
            .reset_index()
        )
        summary.columns = ["Risiko", "Rata-rata", "Median", "Standar Deviasi"]
        st.markdown("##### Ringkasan statistik")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        low_mean = summary.loc[summary["Risiko"] == "Low", "Rata-rata"]
        high_mean = summary.loc[summary["Risiko"] == "High", "Rata-rata"]
        if not low_mean.empty and not high_mean.empty:
            difference = high_mean.iloc[0] - low_mean.iloc[0]
            direction = "lebih tinggi" if difference >= 0 else "lebih rendah"
            st.markdown(
                f"""
                <div class="insight-box">
                    Pada data terpilih, rata-rata <b>{chart_variable}</b> pada kelompok
                    risiko tinggi adalah <b>{abs(difference):.2f} {direction}</b>
                    dibandingkan kelompok risiko rendah. Temuan ini menunjukkan pola
                    deskriptif dan tidak langsung membuktikan hubungan sebab-akibat.
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Korelasi antarvariabel numerik")
    correlation_columns = [
        "Pre_Semester_GPA",
        "Weekly_GenAI_Hours",
        "Tool_Diversity",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
    ]
    corr = filtered[correlation_columns].corr()
    labels = [
        "GPA Awal",
        "Jam GenAI",
        "Keragaman Tool",
        "Belajar Tradisional",
        "Ketergantungan AI",
        "Kecemasan",
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[[0, "#2563EB"], [0.5, "#FFFFFF"], [1, "#EF4444"]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} dengan %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Matriks Korelasi Pearson")
    st.plotly_chart(style_figure(fig, 520), use_container_width=True)

    with st.expander("Lihat data hasil filter"):
        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
        st.caption("Tabel dibatasi maksimal 500 baris agar aplikasi tetap ringan.")


def build_prediction_input(values):
    genai_ratio = values["Weekly_GenAI_Hours"] / (
        values["Traditional_Study_Hours"] + 1
    )
    if values["Weekly_GenAI_Hours"] <= 5:
        usage_category = "Low"
    elif values["Weekly_GenAI_Hours"] <= 15:
        usage_category = "Moderate"
    else:
        usage_category = "High"

    values["GenAI_Study_Ratio"] = genai_ratio
    values["GenAI_Usage_Category"] = usage_category
    return pd.DataFrame([values])


def risk_recommendations(risk):
    if risk == "High":
        return [
            "Evaluasi kembali beban belajar dan frekuensi penggunaan GenAI.",
            "Seimbangkan penggunaan AI dengan proses belajar mandiri.",
            "Pertimbangkan konsultasi dengan dosen wali atau layanan konseling kampus.",
        ]
    if risk == "Medium":
        return [
            "Pantau perubahan kecemasan dan ketergantungan terhadap AI.",
            "Tetapkan batas waktu penggunaan GenAI untuk aktivitas akademik.",
            "Pertahankan jadwal istirahat dan belajar tradisional yang konsisten.",
        ]
    return [
        "Pertahankan pola belajar dan penggunaan GenAI yang seimbang.",
        "Gunakan GenAI sebagai alat bantu, bukan sebagai pengganti proses berpikir.",
        "Lakukan pemantauan berkala jika pola belajar atau beban akademik berubah.",
    ]


def render_prediction(df):
    page_header(
        "Prediction & Analysis",
        "Prediksi Risiko Burnout",
        "Masukkan profil dan kebiasaan belajar mahasiswa. Model akan memberikan kelas risiko serta probabilitas prediksi secara real-time.",
    )

    model, encoder = load_artifacts()
    if model is None or encoder is None:
        st.warning(
            "File model belum ditemukan. Letakkan `burnout_model.pkl` "
            "dan `label_encoder.pkl` dalam folder yang sama dengan `app.py`."
        )

    default = {
        "Pre_Semester_GPA": 3.21,
        "Weekly_GenAI_Hours": 5.8,
        "Tool_Diversity": 3,
        "Traditional_Study_Hours": 11.18,
        "Perceived_AI_Dependency": 3,
        "Anxiety_Level_During_Exams": 4,
    }
    if df is not None:
        for key in default:
            default[key] = float(df[key].median())

    with st.form("prediction_form"):
        st.markdown(
            """
            <div class="insight-box" style="margin-bottom: 1rem;">
                Isi berdasarkan kondisi mahasiswa dalam satu semester terakhir.
                Arahkan kursor ke ikon <b></b> pada setiap pertanyaan untuk melihat
                penjelasan lebih lengkap.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Faktor utama")
        st.caption(
            "Enam isian berikut merupakan informasi utama yang perlu disesuaikan "
            "dengan kondisi mahasiswa."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            pre_gpa = st.number_input(
                "IPK/GPA sebelum semester (0,00-4,00)",
                min_value=0.0,
                max_value=4.0,
                value=float(default["Pre_Semester_GPA"]),
                step=0.01,
                help="Masukkan IPK terakhir sebelum semester yang sedang dianalisis. Contoh: 3,22.",
            )
            genai_hours = st.slider(
                "Penggunaan GenAI per minggu (jam)",
                0.0,
                40.0,
                float(default["Weekly_GenAI_Hours"]),
                0.1,
                help="Perkirakan total jam menggunakan ChatGPT, Gemini, Copilot, atau AI lainnya dalam satu minggu.",
            )
        with c2:
            traditional_hours = st.slider(
                "Belajar tanpa bantuan AI per minggu (jam)",
                1.0,
                36.0,
                float(default["Traditional_Study_Hours"]),
                0.1,
                help="Total waktu membaca, mencatat, berdiskusi, dan mengerjakan tugas tanpa bantuan GenAI.",
            )
            dependency = st.slider(
                "Ketergantungan terhadap AI (1-10)",
                1,
                10,
                int(default["Perceived_AI_Dependency"]),
                help="1 berarti hampir tidak bergantung; 10 berarti sangat sulit belajar atau mengerjakan tugas tanpa AI.",
            )
        with c3:
            anxiety = st.slider(
                "Kecemasan saat ujian (1-10)",
                1,
                10,
                int(default["Anxiety_Level_During_Exams"]),
                help="1 berarti sangat tenang; 10 berarti sangat cemas ketika menghadapi ujian.",
            )
            tool_diversity = st.slider(
                "Jumlah aplikasi AI yang digunakan",
                1,
                5,
                int(default["Tool_Diversity"]),
                help="Contoh: memakai ChatGPT, Gemini, dan Copilot berarti memilih nilai 3.",
            )

        with st.expander("Data pendukung (opsional untuk disesuaikan)", expanded=False):
            st.caption(
                "Bagian ini tetap dibutuhkan model, tetapi sudah memiliki nilai "
                "bawaan. Ubah hanya jika ingin hasil yang lebih sesuai dengan profil."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                major = st.selectbox(
                    "Bidang atau kategori jurusan",
                    ["STEM", "Business", "Medical", "Humanities", "Arts"],
                    format_func={
                        "STEM": "STEM - Sains dan Teknologi",
                        "Business": "Business - Bisnis dan Manajemen",
                        "Medical": "Medical - Kesehatan",
                        "Humanities": "Humanities - Sosial dan Humaniora",
                        "Arts": "Arts - Seni dan Desain",
                    }.get,
                    help="Pilih kategori yang paling sesuai dengan program studi mahasiswa.",
                )
                year = st.selectbox(
                    "Tingkat atau tahun studi",
                    ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
                    format_func={
                        "Freshman": "Tahun pertama",
                        "Sophomore": "Tahun kedua",
                        "Junior": "Tahun ketiga",
                        "Senior": "Tahun keempat/akhir",
                        "Graduate": "Pascasarjana",
                    }.get,
                    help="Pilih tingkat studi mahasiswa saat ini.",
                )
            with c2:
                primary_use = st.selectbox(
                    "Tujuan utama menggunakan GenAI",
                    [
                        "Ideation",
                        "Summarizing_Reading",
                        "Copywriting/Drafting",
                        "Debugging/Troubleshooting",
                        "Direct_Answer_Generation",
                    ],
                    format_func={
                        "Ideation": "Mencari ide",
                        "Summarizing_Reading": "Meringkas bacaan",
                        "Copywriting/Drafting": "Menyusun tulisan",
                        "Debugging/Troubleshooting": "Debugging",
                        "Direct_Answer_Generation": "Mendapatkan jawaban langsung",
                    }.get,
                    help="Pilih aktivitas yang paling sering dilakukan menggunakan GenAI.",
                )
                prompt_skill = st.selectbox(
                    "Kemampuan menyusun prompt",
                    ["Beginner", "Intermediate", "Advanced"],
                    format_func={
                        "Beginner": "Pemula",
                        "Intermediate": "Menengah",
                        "Advanced": "Mahir",
                    }.get,
                    help="Nilai kemampuan memberikan instruksi yang jelas kepada AI.",
                )
            with c3:
                institutional_policy = st.selectbox(
                    "Kebijakan kampus mengenai AI",
                    ["Allowed_With_Citation", "Actively_Encouraged", "Strict_Ban"],
                    format_func={
                        "Allowed_With_Citation": "Diizinkan dengan sumber",
                        "Actively_Encouraged": "Didukung kampus",
                        "Strict_Ban": "Dilarang",
                    }.get,
                    help="Pilih aturan kampus terkait penggunaan GenAI.",
                )
                paid_subscription = st.toggle(
                    "Menggunakan layanan AI berbayar",
                    help="Aktifkan jika menggunakan paket AI berbayar.",
                )

        submitted = st.form_submit_button("Analisis Risiko Burnout")

    if not submitted:
        st.markdown(
            """
            <div class="insight-box">
                Hasil prediksi merupakan dukungan analisis berbasis data dan bukan
                diagnosis kesehatan mental. Interpretasikan hasil bersama kondisi
                nyata mahasiswa.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    values = {
        "Major_Category": major,
        "Year_of_Study": year,
        "Pre_Semester_GPA": pre_gpa,
        "Weekly_GenAI_Hours": genai_hours,
        "Primary_Use_Case": primary_use,
        "Prompt_Engineering_Skill": prompt_skill,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid_subscription,
        "Traditional_Study_Hours": traditional_hours,
        "Perceived_AI_Dependency": dependency,
        "Institutional_Policy": institutional_policy,
        "Anxiety_Level_During_Exams": anxiety,
    }
    prediction_data = build_prediction_input(values)

    if model is None or encoder is None:
        st.error("Prediksi belum dapat dijalankan karena file model belum tersedia.")
        with st.expander("Lihat data yang akan dikirim ke model"):
            st.dataframe(prediction_data, use_container_width=True, hide_index=True)
        return

    try:
        encoded_prediction = model.predict(prediction_data)[0]
        predicted_risk = str(encoder.inverse_transform([encoded_prediction])[0])

        probability_map = {}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(prediction_data)[0]
            model_classes = model.classes_
            decoded_classes = encoder.inverse_transform(model_classes.astype(int))
            probability_map = dict(zip(decoded_classes, probabilities))

        section_title("Hasil analisis")
        risk_color = RISK_COLORS.get(predicted_risk, COLORS["blue"])
        result_left, result_right = st.columns([0.85, 1.15])

        with result_left:
            confidence = probability_map.get(predicted_risk)
            confidence_text = (
                f"{confidence * 100:.1f}%"
                if confidence is not None
                else "Tidak tersedia"
            )
            st.markdown(
                f"""
                <div class="result-card" style="border-top: 5px solid {risk_color};">
                    <div class="metric-label">TINGKAT RISIKO</div>
                    <div style="font-size: 2.4rem; font-weight: 700; color: {risk_color};
                                margin: 0.35rem 0;">{predicted_risk}</div>
                    <div style="color: #64748B;">Keyakinan model: {confidence_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_right:
            if probability_map:
                probability_df = pd.DataFrame(
                    {
                        "Risiko": ["Low", "Medium", "High"],
                        "Probabilitas": [
                            probability_map.get("Low", 0),
                            probability_map.get("Medium", 0),
                            probability_map.get("High", 0),
                        ],
                    }
                )
                fig = px.bar(
                    probability_df,
                    x="Probabilitas",
                    y="Risiko",
                    orientation="h",
                    color="Risiko",
                    color_discrete_map=RISK_COLORS,
                    text=probability_df["Probabilitas"].map(lambda x: f"{x * 100:.1f}%"),
                    title="Probabilitas Setiap Kelas",
                )
                fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Probabilitas")
                fig.update_yaxes(title="")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_figure(fig, 280), use_container_width=True)

        section_title("Rekomendasi awal")
        recommendations = risk_recommendations(predicted_risk)
        for index, recommendation in enumerate(recommendations, start=1):
            st.markdown(f"**{index}.** {recommendation}")

        with st.expander("Detail input dan fitur hasil rekayasa"):
            st.dataframe(prediction_data, use_container_width=True, hide_index=True)
            st.caption(
                "GenAI_Study_Ratio dan GenAI_Usage_Category dihitung otomatis "
                "menggunakan aturan yang sama seperti pada notebook."
            )
    except Exception as error:
        st.error(
            "Prediksi gagal karena struktur input aplikasi tidak sama dengan model "
            f"yang disimpan. Detail error: {error}"
        )
        st.info(
            "Pastikan model dilatih menggunakan daftar fitur yang sama dengan kode "
            "Data Preparation dan seluruh preprocessing disimpan di dalam Pipeline."
        )


def render_documentation(df):
    page_header(
        "About & Documentation",
        "Dokumentasi Proyek",
        "Penjelasan metode, model, evaluasi, cara penggunaan aplikasi, dan informasi tim.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Metodologi", "Model & Evaluasi", "Cara Penggunaan", "Informasi Tim"]
    )

    with tab1:
        st.markdown("### Metode CRISP-DM")
        stages = [
            ("1. Business Understanding", "Menentukan masalah, tujuan, manfaat, dan ruang lingkup prediksi burnout."),
            ("2. Data Understanding", "Mengidentifikasi 50.000 record, 16 variabel, distribusi target, dan pola awal data."),
            ("3. Data Preparation", "Memeriksa kualitas data, memilih fitur, membuat fitur baru, encoding, scaling, dan train-test split."),
            ("4. Modeling", "Membandingkan Logistic Regression, Random Forest, dan Gradient Boosting."),
            ("5. Evaluation", "Menilai model menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix."),
            ("6. Deployment", "Menerapkan model terpilih dalam aplikasi Streamlit untuk analisis dan prediksi real-time."),
        ]
        for title, description in stages:
            st.markdown(
                f'<div class="info-card" style="margin-bottom: 0.65rem;"><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("### Penjelasan Model")
        model, encoder = load_artifacts()
        final_model_name = get_model_name(model)

        st.write(
            "Masalah diselesaikan sebagai klasifikasi multikelas dengan target "
            "`Burnout_Risk_Level`: Low, Medium, dan High. Model final dipilih "
            "berdasarkan hasil evaluasi pada data uji, terutama macro F1-score."
        )

        c1, c2 = st.columns(2)
        with c1:
            metric_card(
                "Model Final",
                final_model_name,
                "Hasil perbandingan dan tuning",
            )
        with c2:
            deployment_status = (
                "Siap digunakan"
                if model is not None and encoder is not None
                else "Belum tersedia"
            )
            metric_card(
                "Status Deployment",
                deployment_status,
                "Model dan encoder",
            )

        st.markdown("#### Fitur yang digunakan")
        feature_table = pd.DataFrame(
            [
                ["Major_Category", "Kategorikal", "Kategori bidang studi"],
                ["Year_of_Study", "Kategorikal", "Tingkat tahun studi"],
                ["Pre_Semester_GPA", "Numerik", "GPA sebelum semester"],
                ["Weekly_GenAI_Hours", "Numerik", "Jam penggunaan GenAI per minggu"],
                ["Primary_Use_Case", "Kategorikal", "Tujuan utama penggunaan GenAI"],
                ["Prompt_Engineering_Skill", "Kategorikal", "Kemampuan menyusun prompt"],
                ["Tool_Diversity", "Numerik", "Jumlah jenis tool AI"],
                ["Paid_Subscription", "Boolean", "Status langganan AI berbayar"],
                ["Traditional_Study_Hours", "Numerik", "Jam belajar tradisional"],
                ["Perceived_AI_Dependency", "Numerik", "Tingkat ketergantungan AI"],
                ["Institutional_Policy", "Kategorikal", "Kebijakan institusi"],
                ["Anxiety_Level_During_Exams", "Numerik", "Kecemasan saat ujian"],
                ["GenAI_Study_Ratio", "Rekayasa fitur", "Rasio penggunaan GenAI dan belajar tradisional"],
                ["GenAI_Usage_Category", "Rekayasa fitur", "Kategori intensitas penggunaan GenAI"],
            ],
            columns=["Variabel", "Jenis", "Keterangan"],
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.markdown("#### Metrik evaluasi")
        evaluation_df = load_evaluation()
        if evaluation_df is not None and not evaluation_df.empty:
            display_evaluation = evaluation_df.copy()
            numeric_columns = display_evaluation.select_dtypes(include=np.number).columns
            display_evaluation[numeric_columns] = display_evaluation[
                numeric_columns
            ].round(4)
            st.dataframe(
                display_evaluation,
                use_container_width=True,
                hide_index=True,
            )

            f1_column = next(
                (
                    column
                    for column in display_evaluation.columns
                    if column.lower().replace("-", " ").replace("_", " ")
                    in {"f1 macro", "f1 score macro"}
                ),
                None,
            )
            if f1_column and "Model" in display_evaluation.columns:
                best_row = display_evaluation.loc[
                    display_evaluation[f1_column].idxmax()
                ]
                st.success(
                    f"Model dengan {f1_column} tertinggi pada data testing adalah "
                    f"{best_row['Model']} dengan skor {best_row[f1_column]:.4f}."
                )
        else:
            st.info(
                "File `model_evaluation.csv` belum tersedia. Ekspor tabel hasil "
                "perbandingan model dari notebook agar metrik tampil di bagian ini."
            )

    with tab3:
        st.markdown("### Cara Menggunakan Aplikasi")
        st.markdown(
            """
            1. Buka menu **Dashboard EDA** untuk mengeksplorasi data.
            2. Atur filter jurusan, tahun studi, tingkat risiko, dan GPA.
            3. Buka menu **Prediksi Burnout**.
            4. Isi seluruh profil akademik dan kebiasaan belajar.
            5. Tekan tombol **Analisis Risiko Burnout**.
            6. Baca kelas risiko, probabilitas, dan rekomendasi awal.
            """
        )
        st.markdown("### Menjalankan Secara Lokal")
        st.code(
            "pip install -r requirements.txt\nstreamlit run app.py",
            language="bash",
        )

    with tab4:
        st.markdown("### Informasi Tim")
        st.markdown(
            """
            <div class="info-card" style="max-width: 720px;">
                <div class="metric-label">NAMA TIM</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #102A43;
                            margin: 0.25rem 0 1.1rem;">WhatIf</div>
                <div class="metric-label">ANGGOTA</div>
                <div style="font-size: 1rem; font-weight: 600; color: #334E68;
                            margin: 0.25rem 0 1.1rem;">Ihwan Fajar Maulana</div>
                <div class="metric-label">PROGRAM STUDI</div>
                <div style="font-size: 1rem; font-weight: 600; color: #334E68;
                            margin-top: 0.25rem;">S1 Sistem Informasi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if df is not None:
            st.caption(
                f"Dataset aktif: {len(df):,} record dan {df.shape[1]} variabel."
            )


def main():
    apply_css()
    df, data_path = load_data()
    page = render_sidebar()

    if page == "Beranda":
        render_home(df)
    elif page == "Dashboard EDA":
        render_eda(df)
    elif page == "Prediksi Burnout":
        render_prediction(df)
    else:
        render_documentation(df)

    st.markdown(
        """
        <div class="footer">
            Student Burnout Intelligence • Education & Risk Prediction • CRISP-DM
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
