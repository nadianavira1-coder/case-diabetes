import os
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
# KONFIGURASI
# ======================================================
st.set_page_config(page_title="SPK Risiko Diabetes - KNN", page_icon="🩺", layout="wide")

FILE_DATA = "dataset.csv"
FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
KOLOM = ["Pasien", "Jenis Kelamin"] + FITUR + ["Kelas"]

DATA_AWAL = [
    ["P01", "-", 25, 21.5, 90, 110, 0, 0],
    ["P02", "-", 30, 23.1, 95, 115, 0, 0],
    ["P03", "-", 35, 25.0, 105, 120, 0, 0],
    ["P04", "-", 40, 27.5, 120, 125, 1, 1],
    ["P05", "-", 45, 30.2, 140, 135, 1, 1],
    ["P06", "-", 50, 31.5, 155, 140, 1, 1],
    ["P07", "-", 55, 33.0, 165, 145, 1, 1],
    ["P08", "-", 28, 22.5, 92, 112, 0, 0],
    ["P09", "-", 38, 26.8, 115, 122, 0, 0],
    ["P10", "-", 48, 29.5, 145, 130, 1, 1],
]

LABEL = {0: "Risiko Rendah", 1: "Risiko Tinggi"}


# ======================================================
# STYLE (NUANSA BIRU SOFT)
# ======================================================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #eaf4ff 0%, #f6faff 100%); }
    h1, h2, h3, h4 { color: #1f4e79 !important; }
    .hero {
        background: linear-gradient(135deg, #8ec5fc 0%, #b7dcff 100%);
        padding: 26px 30px; border-radius: 20px; margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(80,140,200,.25);
    }
    .hero h1 { margin: 0; color: #12395f !important; font-size: 2rem; }
    .hero p  { margin: 6px 0 0 0; color: #1f4e79; }
    .card {
        background: #ffffff; border-radius: 16px; padding: 18px 22px;
        box-shadow: 0 3px 12px rgba(80,140,200,.15); margin-bottom: 14px;
        border-left: 6px solid #6fb1f2;
    }
    .step {
        background: #dcecff; color: #1f4e79; font-weight: 700;
        padding: 8px 16px; border-radius: 10px; margin: 18px 0 8px 0;
        display: inline-block;
    }
    .hasil-rendah, .hasil-tinggi {
        border-radius: 18px; padding: 24px 28px; margin-top: 10px;
        font-size: 1.05rem; box-shadow: 0 6px 18px rgba(80,140,200,.2);
    }
    .hasil-rendah { background: #e3f6ec; border: 2px solid #6cc08b; color: #1d5a37; }
    .hasil-tinggi { background: #ffe9e9; border: 2px solid #f09595; color: #7a2323; }
    .hasil-rendah h2, .hasil-tinggi h2 { margin: 0 0 8px 0; color: inherit !important; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: #4f9bea; color: white; border: none; border-radius: 12px;
        padding: 10px 26px; font-weight: 600;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #3a86d8; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🩺 Sistem Pendukung Keputusan Risiko Diabetes</h1>
      <p>Klasifikasi pasien dengan algoritma K-Nearest Neighbors (KNN)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# DATA
# ======================================================
def muat_data():
    if not os.path.exists(FILE_DATA):
        pd.DataFrame(DATA_AWAL, columns=KOLOM).to_csv(FILE_DATA, index=False)
    return pd.read_csv(FILE_DATA)


def simpan_data(df):
    df.to_csv(FILE_DATA, index=False)


def id_berikutnya(df):
    angka = pd.to_numeric(df["Pasien"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").max()
    angka = 0 if pd.isna(angka) else int(angka)
    return f"P{angka + 1:02d}"


# ======================================================
# ALGORITMA KNN
# ======================================================
def hitung_knn(df, x_baru, k):
    X = df[FITUR].astype(float)
    mn, mx = X.min(), X.max()
    rentang = (mx - mn).replace(0, 1)

    X_norm = (X - mn) / rentang
    baru = pd.Series(x_baru, index=FITUR, dtype=float)
    baru_norm = (baru - mn) / rentang

    selisih_kuadrat = (X_norm - baru_norm) ** 2
    jarak = np.sqrt(selisih_kuadrat.sum(axis=1))

    hasil = pd.DataFrame({"Pasien": df["Pasien"]})
    for f in FITUR:
        hasil[f"({f})²"] = selisih_kuadrat[f].round(4)
    hasil["Jarak Euclidean"] = jarak.round(4)
    hasil["Kelas"] = df["Kelas"].map(LABEL)
    hasil["_kelas"] = df["Kelas"]
    hasil = hasil.sort_values("Jarak Euclidean").reset_index(drop=True)
    hasil.insert(0, "Rank", hasil.index + 1)

    tetangga = hasil.head(k)
    suara = tetangga["_kelas"].value_counts()
    n0, n1 = int(suara.get(0, 0)), int(suara.get(1, 0))
    if n0 == n1:  # seri -> ikut tetangga terdekat
        prediksi = int(tetangga.iloc[0]["_kelas"])
    else:
        prediksi = 1 if n1 > n0 else 0

    return {
        "min": mn, "max": mx, "baru_norm": baru_norm,
        "hasil": hasil, "tetangga": tetangga,
        "n0": n0, "n1": n1, "prediksi": prediksi,
    }


# ======================================================
# HALAMAN
# ======================================================
df = muat_data()
tab1, tab2 = st.tabs(["🔍 Prediksi Pasien", "📋 Dataset"])

# ---------------- TAB PREDIKSI ----------------
with tab1:
    st.markdown("### Input Data Pasien")
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        nama = c1.text_input("Nama Pasien", placeholder="Contoh: Budi")
        jk = c2.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        c3, c4, c5 = st.columns(3)
        usia = c3.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
        bmi = c4.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")
        glukosa = c5.number_input("Glukosa (mg/dL)", min_value=40, max_value=400, value=130, step=1)

        c6, c7, c8 = st.columns(3)
        tensi = c6.number_input("Tekanan Darah (sistolik)", min_value=60, max_value=250, value=128, step=1)
        riwayat = c7.selectbox("Riwayat Diabetes Keluarga", ["Tidak Ada (0)", "Ada (1)"], index=1)
        opsi_k = list(range(1, len(df) + 1, 2))
        k = c8.select_slider("Nilai K", options=opsi_k, value=3 if 3 in opsi_k else opsi_k[0])

        proses = st.form_submit_button("🔎 Hitung Prediksi")

    if proses:
        if not nama.strip():
            st.warning("Nama pasien wajib diisi.")
        else:
            rwy = 1 if riwayat.startswith("Ada") else 0
            x_baru = [usia, bmi, glukosa, tensi, rwy]
            st.session_state["hasil"] = {
                "nama": nama.strip(), "jk": jk, "x": x_baru, "k": k,
                "knn": hitung_knn(df, x_baru, k),
            }

    if "hasil" in st.session_state:
        r = st.session_state["hasil"]
        knn = r["knn"]
        k_pakai = r["k"]

        st.markdown("---")
        st.markdown("## ⚙️ Proses Perhitungan KNN")

        st.markdown('<div class="step">Langkah 1 — Data Pasien Baru</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([[r["nama"], r["jk"]] + r["x"]], columns=["Nama", "Jenis Kelamin"] + FITUR),
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Nilai K yang digunakan: **{k_pakai}**")

        st.markdown('<div class="step">Langkah 2 — Normalisasi Min-Max</div>', unsafe_allow_html=True)
        st.latex(r"x' = \frac{x - x_{min}}{x_{max} - x_{min}}")
        norm_df = pd.DataFrame({
            "Fitur": FITUR,
            "Min": knn["min"].values,
            "Max": knn["max"].values,
            "Nilai Baru": r["x"],
            "Hasil Normalisasi": knn["baru_norm"].round(4).values,
        })
        st.dataframe(norm_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="step">Langkah 3 — Jarak Euclidean ke Setiap Data Latih</div>', unsafe_allow_html=True)
        st.latex(r"d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}")
        tampil = knn["hasil"].drop(columns=["_kelas"])
        st.dataframe(tampil, hide_index=True, use_container_width=True)

        st.markdown(f'<div class="step">Langkah 4 — {k_pakai} Tetangga Terdekat</div>', unsafe_allow_html=True)
        st.dataframe(
            knn["tetangga"][["Rank", "Pasien", "Jarak Euclidean", "Kelas"]],
            hide_index=True, use_container_width=True,
        )

        st.markdown('<div class="step">Langkah 5 — Voting Mayoritas</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        v1.metric("Suara Risiko Rendah (0)", knn["n0"])
        v2.metric("Suara Risiko Tinggi (1)", knn["n1"])

        st.markdown("## ✅ Kesimpulan")
        pred = knn["prediksi"]
        kelas_css = "hasil-tinggi" if pred == 1 else "hasil-rendah"
        ikon = "⚠️" if pred == 1 else "💙"
        saran = (
            "Disarankan melakukan pemeriksaan lanjutan dan konsultasi dengan dokter."
            if pred == 1
            else "Pertahankan pola hidup sehat dan lakukan pemeriksaan rutin."
        )
        st.markdown(
            f"""
            <div class="{kelas_css}">
              <h2>{ikon} {LABEL[pred]}</h2>
              Pasien <b>{r['nama']}</b> ({r['jk']}, usia {r['x'][0]} tahun) diprediksi
              memiliki <b>{LABEL[pred]}</b> diabetes berdasarkan {k_pakai} tetangga terdekat
              ({knn['n0']} suara Rendah, {knn['n1']} suara Tinggi).<br><br>
              <i>{saran}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("➕ Simpan pasien ini ke dataset"):
            baris = pd.DataFrame(
                [[id_berikutnya(df), r["jk"]] + r["x"] + [pred]], columns=KOLOM
            )
            simpan_data(pd.concat([df, baris], ignore_index=True))
            del st.session_state["hasil"]
            st.success("Pasien berhasil ditambahkan ke dataset.")
            st.rerun()

# ---------------- TAB DATASET ----------------
with tab2:
    st.markdown("### Dataset Pasien")
    tampil = df.copy()
    tampil["Kelas"] = tampil["Kelas"].map(LABEL)
    st.dataframe(tampil, hide_index=True, use_container_width=True)
    st.caption(f"Total data: {len(df)} pasien")

    st.markdown("### ➕ Tambah Data Baru")
    with st.form("form_tambah", clear_on_submit=True):
        a1, a2, a3 = st.columns(3)
        t_nama = a1.text_input("ID / Nama Pasien", value=id_berikutnya(df))
        t_jk = a2.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key="t_jk")
        t_kelas = a3.selectbox("Kelas", ["Risiko Rendah (0)", "Risiko Tinggi (1)"])

        b1, b2, b3, b4, b5 = st.columns(5)
        t_usia = b1.number_input("Usia", 1, 120, 30)
        t_bmi = b2.number_input("BMI", 10.0, 60.0, 24.0, 0.1, format="%.1f")
        t_glu = b3.number_input("Glukosa", 40, 400, 100)
        t_tensi = b4.number_input("Tekanan Darah", 60, 250, 120)
        t_rwy = b5.selectbox("Riwayat Keluarga", ["Tidak Ada (0)", "Ada (1)"])

        if st.form_submit_button("Simpan ke Dataset"):
            baris = pd.DataFrame(
                [[
                    t_nama.strip() or id_berikutnya(df), t_jk,
                    t_usia, t_bmi, t_glu, t_tensi,
                    1 if t_rwy.startswith("Ada") else 0,
                    1 if t_kelas.endswith("(1)") else 0,
                ]],
                columns=KOLOM,
            )
            simpan_data(pd.concat([df, baris], ignore_index=True))
            st.success("Data berhasil ditambahkan.")
            st.rerun()

    st.markdown("### 🗑️ Kelola Data")
    d1, d2 = st.columns(2)
    hapus = d1.selectbox("Hapus pasien", ["- pilih -"] + df["Pasien"].astype(str).tolist())
    if d1.button("Hapus"):
        if hapus != "- pilih -":
            simpan_data(df[df["Pasien"].astype(str) != hapus])
            st.rerun()
    if d2.button("♻️ Reset ke 10 data awal"):
        pd.DataFrame(DATA_AWAL, columns=KOLOM).to_csv(FILE_DATA, index=False)
        st.session_state.pop("hasil", None)
        st.rerun()

st.caption("Catatan: Jenis kelamin dicatat sebagai identitas pasien, tidak ikut dihitung dalam jarak KNN karena tidak ada di dataset awal.")
