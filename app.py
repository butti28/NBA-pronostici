import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Pronostici NBA", layout="centered")
st.title("Pronostici NBA di oggi")
st.caption("Stime statistiche indicative. Non garantiscono vincite. Controlla sempre gli infortuni.")

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
GIORNI_STORICO = 120


def chiama(data_str):
    r = requests.get(BASE, params={"dates": data_str, "limit": 100}, timeout=20)
    r.raise_for_status()
    return r.json().get("events", [])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def storico(fine_str, giorni):
    fine = datetime.strptime(fine_str, "%Y%m%d")
    righe = []
    for i in range(1, giorni + 1):
        giorno = fine - timedelta(days=i)
        try:
            eventi = chiama(giorno.strftime("%Y%m%d"))
        except Exception:
            continue
        for e in eventi:
            comp = e["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            sq = comp["competitors"]
            casa = next(c for c in sq if c["homeAway"] == "home")
            trasf = next(c for c in sq if c["homeAway"] == "away")
            righe.append({
                "data": giorno,
                "casa": casa["team"]["displayName"],
                "trasf": trasf["team"]["displayName"],
                "p_casa": float(casa["score"]),
                "p_trasf": float(trasf["score"]),
            })
    return pd.DataFrame(righe)


def tabella_squadre(df):
    a = df.rename(columns={"casa": "squadra", "p_casa": "fatti", "p_trasf": "subiti"})[["data", "squadra", "fatti", "subiti"]]
    b = df.rename(columns={"trasf": "squadra", "p_trasf": "fatti", "p_casa": "subiti"})[["data", "squadra", "fatti", "subiti"]]
    t = pd.concat([a, b]).sort_values(["squadra", "data"]).copy()
    t["vinta"] = (t["fatti"] > t["subiti"]).astype(int)
    t["diff"] = t["fatti"] - t["subiti"]
    return t


def stato(t, squadra, giorno):
    r = t[(t["squadra"] == squadra) & (t["data"] < giorno)].tail(10)
    if len(r) < 3:
        return None
    ultima = t[(t["squadra"] == squadra) & (t["data"] < giorno)]["data"].max()
    return {
        "forma": r["vinta"].mean(),
        "fatti": r["fatti"].mean(),
        "subiti": r["subiti"].mean(),
        "diff": r["diff"].mean(),
        "riposo": min((giorno - ultima).days, 5),
    }


def vettore(a, b):
    return [a["forma"] - b["forma"], a["fatti"] - b["fatti"], a["diff"] - b["diff"], a["riposo"] - b["riposo"]]


@st.cache_resource(show_spinner=False)
def addestra(fine_str, giorni):
    df = storico(fine_str, giorni)
    if df.empty or len(df) < 80:
        return None
    t = tabella_squadre(df)
    X, y = [], []
    for _, g in df.iterrows():
        a, b = stato(t, g["casa"], g["data"]), stato(t, g["trasf"], g["data"])
        if a and b:
            X.append(vettore(a, b))
            y.append(int(g["p_casa"] > g["p_trasf"]))
    if len(X) < 60:
        return None
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    return m


oggi = datetime.now(ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
data_scelta = st.date_input("Giorno", value=oggi.date())
giorno = datetime(data_scelta.year, data_scelta.month, data_scelta.day)
giorno_str = giorno.strftime("%Y%m%d")

try:
    with st.spinner("Carico calendario e statistiche (la prima volta ci vuole un po')..."):
        eventi = chiama(giorno_str)
        df_storico = storico(giorno_str, GIORNI_STORICO)
        modello = addestra(giorno_str, GIORNI_STORICO)
except Exception as e:
    st.error("Non riesco a scaricare i dati in questo momento. Riprova tra poco.")
    st.caption(f"Dettaglio tecnico: {e}")
    st.stop()

if not eventi:
    st.info("Nessuna partita NBA in questa data.")
    st.stop()
if modello is None or df_storico.empty:
    st.warning("Dati storici insufficienti per calcolare i pronostici.")
    st.stop()

tab = tabella_squadre(df_storico)
st.subheader(f"{len(eventi)} partite")

for e in eventi:
    comp = e["competitions"][0]
    sq = comp["competitors"]
    c = next(x for x in sq if x["homeAway"] == "home")
    t_ = next(x for x in sq if x["homeAway"] == "away")
    casa, trasf = c["team"]["displayName"], t_["team"]["displayName"]
    stato_gara = comp["status"]["type"].get("shortDetail", "")
    a, b = stato(tab, casa, giorno), stato(tab, trasf, giorno)

    if not a or not b:
        with st.expander(f"{trasf} @ {casa}  |  {stato_gara}"):
            st.write("Dati insufficienti per questa partita.")
        continue

    p_casa = float(modello.predict_proba([vettore(a, b)])[0][1])
    fav = casa if p_casa >= 0.5 else trasf
    p_fav = max(p_casa, 1 - p_casa)

    with st.expander(f"{trasf} @ {casa}  |  {stato_gara}  |  Pronostico: {fav} {p_fav * 100:.0f}%"):
        k1, k2 = st.columns(2)
        k1.metric(casa + " (casa)", f"{p_casa * 100:.1f}%")
        k2.metric(trasf + " (trasferta)", f"{(1 - p_casa) * 100:.1f}%")
        st.write(f"Quota equa {casa}: {1 / p_casa:.2f}  |  {trasf}: {1 / (1 - p_casa):.2f}")

        pc = (a["fatti"] + b["subiti"]) / 2
        pt = (b["fatti"] + a["subiti"]) / 2
        st.write(f"Punti stimati: {casa} {pc:.0f}  |  {trasf} {pt:.0f}  |  Totale circa {pc + pt:.0f}")

        quote = comp.get("odds")
        if quote:
            q = quote[0]
            st.markdown("**Quote del bookmaker (ESPN)**")
            st.write(f"Fornitore: {q.get('provider', {}).get('name', 'n.d.')}")
            if q.get("details"):
                st.write(f"Handicap: {q['details']}")
            if q.get("overUnder"):
                st.write(f"Totale punti (over/under): {q['overUnder']}")
                diff = (pc + pt) - float(q["overUnder"])
                st.write(f"Il modello stima {abs(diff):.0f} punti in {'più' if diff > 0 else 'meno'} rispetto alla linea.")
        else:
            st.caption("Quote non disponibili per questa partita.")

        for nome, s in [(casa, a), (trasf, b)]:
            leader = next((x for x in next(y for y in sq if y['team']['displayName'] == nome).get('leaders', []) if x.get('name') == 'points'), None)
            if leader and leader.get('leaders'):
                gl = leader['leaders'][0]
                st.write(f"Miglior marcatore {nome}: {gl['athlete']['displayName']}, {gl.get('displayValue', '')}")

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