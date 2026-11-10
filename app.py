import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from nba_api.stats.endpoints import leaguegamelog, scoreboardv2, playergamelog
from nba_api.stats.static import teams
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Pronostici NBA", layout="centered")
st.title("Pronostici NBA di oggi")
st.caption("Stime statistiche indicative. Non garantiscono vincite. Controlla sempre gli infortuni.")

STAGIONI_ADDESTRAMENTO = ["2023-24", "2024-25"]
STAGIONE_CORRENTE = "2025-26"
elenco = teams.get_teams()
nome_da_id = {t["id"]: t["full_name"] for t in elenco}


@st.cache_data(ttl=3600)
def scarica_partite(stagione):
    log = leaguegamelog.LeagueGameLog(
        season=stagione, season_type_all_star="Regular Season"
    )
    df = log.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["VITTORIA"] = (df["WL"] == "W").astype(int)
    return df


def prepara_dati(df):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    df["CASA"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    g = df.groupby("TEAM_ID")
    df["FORMA"] = g["VITTORIA"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    df["PUNTI"] = g["PTS"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    df["DIFF"] = g["PLUS_MINUS"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    df["RIPOSO"] = g["GAME_DATE"].diff().dt.days.clip(upper=5).fillna(3)
    return df.dropna(subset=["FORMA", "PUNTI", "DIFF"])


@st.cache_resource
def addestra_modello():
    pezzi = [prepara_dati(scarica_partite(s)) for s in STAGIONI_ADDESTRAMENTO]
    df = pd.concat(pezzi)
    m = df[df["CASA"] == 1].merge(df[df["CASA"] == 0], on="GAME_ID", suffixes=("_c", "_t"))
    X = pd.DataFrame({
        "forma": m["FORMA_c"] - m["FORMA_t"],
        "punti": m["PUNTI_c"] - m["PUNTI_t"],
        "diff": m["DIFF_c"] - m["DIFF_t"],
        "riposo": m["RIPOSO_c"] - m["RIPOSO_t"],
    })
    modello = LogisticRegression(max_iter=1000)
    modello.fit(X, m["VITTORIA_c"])
    return modello


def stato_squadra(df, team_id, oggi):
    r = df[df["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    ultime = r.tail(10)
    if len(ultime) == 0:
        return None
    riposo = min((oggi - r["GAME_DATE"].max()).days, 5)
    return {
        "forma": ultime["VITTORIA"].mean(),
        "punti": ultime["PTS"].mean(),
        "subiti": (ultime["PTS"] - ultime["PLUS_MINUS"]).mean(),
        "diff": ultime["PLUS_MINUS"].mean(),
        "riposo": riposo,
    }


@st.cache_data(ttl=600)
def partite_di_oggi(data_str):
    sb = scoreboardv2.ScoreboardV2(game_date=data_str)
    return sb.get_data_frames()[0]


@st.cache_data(ttl=3600)
def miglior_marcatore(team_id):
    try:
        log = leaguegamelog.LeagueGameLog(
            season=STAGIONE_CORRENTE,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
        ).get_data_frames()[0]
        sq = log[log["TEAM_ID"] == team_id].sort_values("GAME_DATE")
        ultime_date = sq["GAME_DATE"].drop_duplicates().tail(10)
        sq = sq[sq["GAME_DATE"].isin(ultime_date)]
        medie = sq.groupby("PLAYER_NAME")["PTS"].agg(["mean", "count"])
        medie = medie[medie["count"] >= 5].sort_values("mean", ascending=False)
        return medie.index[0], float(medie.iloc[0]["mean"])
    except Exception:
        return None, None


oggi_it = datetime.now(ZoneInfo("America/New_York"))
data_scelta = st.date_input("Giorno", value=oggi_it.date())
data_str = data_scelta.strftime("%m/%d/%Y")

with st.spinner("Carico calendario e statistiche..."):
    try:
        calendario = partite_di_oggi(data_str)
        modello = addestra_modello()
        dati = scarica_partite(STAGIONE_CORRENTE)
    except Exception as e:
        st.error("Non riesco a scaricare i dati NBA in questo momento. Riprova tra poco.")
        st.stop()

if calendario.empty:
    st.info("Nessuna partita NBA in questa data.")
    st.stop()

data_ts = pd.Timestamp(data_scelta)
st.subheader(f"{len(calendario)} partite")

for _, p in calendario.iterrows():
    id_casa, id_trasf = int(p["HOME_TEAM_ID"]), int(p["VISITOR_TEAM_ID"])
    casa, trasf = nome_da_id[id_casa], nome_da_id[id_trasf]
    a, b = stato_squadra(dati, id_casa, data_ts), stato_squadra(dati, id_trasf, data_ts)
    orario = p.get("GAME_STATUS_TEXT", "")

    if a is None or b is None:
        with st.expander(f"{trasf} @ {casa}  {orario}"):
            st.write("Dati insufficienti per questa partita.")
        continue

    x = pd.DataFrame([{
        "forma": a["forma"] - b["forma"],
        "punti": a["punti"] - b["punti"],
        "diff": a["diff"] - b["diff"],
        "riposo": a["riposo"] - b["riposo"],
    }])
    p_casa = float(modello.predict_proba(x)[0][1])
    favorita = casa if p_casa >= 0.5 else trasf
    p_fav = max(p_casa, 1 - p_casa)

    titolo = f"{trasf} @ {casa}  |  {orario}  |  Favorita: {favorita} {p_fav * 100:.0f}%"
    with st.expander(titolo):
        c1, c2 = st.columns(2)
        c1.metric(casa + " (casa)", f"{p_casa * 100:.1f}%")
        c2.metric(trasf + " (trasferta)", f"{(1 - p_casa) * 100:.1f}%")
        st.write(f"Quota equa {casa}: {1 / p_casa:.2f}   |   Quota equa {trasf}: {1 / (1 - p_casa):.2f}")

        tot = (a["punti"] + b["subiti"]) / 2 + (b["punti"] + a["subiti"]) / 2
        st.write(f"Punti totali stimati: circa {tot:.0f}")
        st.write(f"Punti stimati {casa}: {(a['punti'] + b['subiti']) / 2:.0f}  |  {trasf}: {(b['punti'] + a['subiti']) / 2:.0f}")

        st.markdown("**Miglior marcatore recente**")
        for nome_sq, tid in [(casa, id_casa), (trasf, id_trasf)]:
            gioc, media = miglior_marcatore(tid)
            if gioc:
                st.write(f"{nome_sq}: {gioc}, media {media:.1f} punti nelle ultime partite")

        st.markdown("**Forma recente (ultime 10)**")
        st.write(f"{casa}: {a['forma'] * 10:.0f} vittorie su 10, riposo {a['riposo']} giorni")
        st.write(f"{trasf}: {b['forma'] * 10:.0f} vittorie su 10, riposo {b['riposo']} giorni")

        if p_fav >= 0.65:
            st.success("Favorita netta secondo il modello.")
        elif p_fav >= 0.55:
            st.info("Leggera favorita.")
        else:
            st.warning("Partita equilibrata, da evitare secondo il modello.")
        st.caption("Gioca solo se la quota del bookmaker supera la quota equa. Verifica gli infortuni.")