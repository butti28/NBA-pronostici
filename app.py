import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Pronostici NBA", layout="centered")
st.title("Pronostici NBA personali")
st.caption("Modello statistico indicativo. Non garantisce vincite.")

STAGIONE_ADDESTRAMENTO = ["2023-24", "2024-25"]
STAGIONE_CORRENTE = "2025-26"


@st.cache_data(ttl=3600)
def scarica_partite(stagione):
    log = leaguegamelog.LeagueGameLog(
        season=stagione, season_type_all_star="Regular Season"
    )
    return log.get_data_frames()[0]


def prepara_dati(df):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["VITTORIA"] = (df["WL"] == "W").astype(int)
    df["CASA"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    gruppo = df.groupby("TEAM_ID")
    df["FORMA"] = gruppo["VITTORIA"].transform(
        lambda s: s.shift().rolling(10, min_periods=3).mean()
    )
    df["PUNTI_FATTI"] = gruppo["PTS"].transform(
        lambda s: s.shift().rolling(10, min_periods=3).mean()
    )
    df["DIFF_PUNTI"] = gruppo["PLUS_MINUS"].transform(
        lambda s: s.shift().rolling(10, min_periods=3).mean()
    )
    df["RIPOSO"] = (
        gruppo["GAME_DATE"].diff().dt.days.clip(upper=5).fillna(3)
    )
    return df.dropna(subset=["FORMA", "PUNTI_FATTI", "DIFF_PUNTI"])


def costruisci_confronti(df):
    casa = df[df["CASA"] == 1]
    trasf = df[df["CASA"] == 0]
    m = casa.merge(trasf, on="GAME_ID", suffixes=("_c", "_t"))
    X = pd.DataFrame({
        "forma": m["FORMA_c"] - m["FORMA_t"],
        "punti": m["PUNTI_FATTI_c"] - m["PUNTI_FATTI_t"],
        "diff": m["DIFF_PUNTI_c"] - m["DIFF_PUNTI_t"],
        "riposo": m["RIPOSO_c"] - m["RIPOSO_t"],
    })
    y = m["VITTORIA_c"]
    return X, y


@st.cache_resource
def addestra_modello():
    pezzi = [prepara_dati(scarica_partite(s)) for s in STAGIONE_ADDESTRAMENTO]
    X, y = costruisci_confronti(pd.concat(pezzi))
    modello = LogisticRegression(max_iter=1000)
    modello.fit(X, y)
    return modello


def ultime_statistiche(df, team_id):
    r = df[df["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    ultime = r.tail(10)
    ultima_data = r["GAME_DATE"].max()
    return {
        "forma": (ultime["WL"] == "W").mean(),
        "punti": ultime["PTS"].mean(),
        "diff": ultime["PLUS_MINUS"].mean(),
        "ultima_data": ultima_data,
    }


elenco = teams.get_teams()
nomi = sorted(t["full_name"] for t in elenco)
id_per_nome = {t["full_name"]: t["id"] for t in elenco}

col1, col2 = st.columns(2)
casa = col1.selectbox("Squadra in casa", nomi, index=0)
trasferta = col2.selectbox("Squadra in trasferta", nomi, index=1)
riposo_casa = st.slider("Giorni di riposo squadra in casa", 0, 5, 2)
riposo_trasf = st.slider("Giorni di riposo squadra in trasferta", 0, 5, 2)

if st.button("Calcola pronostico"):
    if casa == trasferta:
        st.warning("Scegli due squadre diverse.")
    else:
        with st.spinner("Scarico i dati e calcolo..."):
            modello = addestra_modello()
            dati = scarica_partite(STAGIONE_CORRENTE)
            dati["GAME_DATE"] = pd.to_datetime(dati["GAME_DATE"])
            dati["VITTORIA"] = (dati["WL"] == "W").astype(int)
            sc = ultime_statistiche(dati, id_per_nome[casa])
            st_ = ultime_statistiche(dati, id_per_nome[trasferta])
            x = pd.DataFrame([{
                "forma": sc["forma"] - st_["forma"],
                "punti": sc["punti"] - st_["punti"],
                "diff": sc["diff"] - st_["diff"],
                "riposo": riposo_casa - riposo_trasf,
            }])
            p_casa = modello.predict_proba(x)[0][1]
        st.subheader("Probabilità di vittoria")
        st.metric(casa, f"{p_casa * 100:.1f}%")
        st.metric(trasferta, f"{(1 - p_casa) * 100:.1f}%")
        st.write(f"Quota equa {casa}: {1 / p_casa:.2f}")
        st.write(f"Quota equa {trasferta}: {1 / (1 - p_casa):.2f}")
        st.info(
            "Gioca solo se la quota del bookmaker è più alta della quota equa. "
            "Controlla sempre gli infortuni prima di giocare."
        )