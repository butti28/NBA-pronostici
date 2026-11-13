import base64
import json
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Pronostici NBA", page_icon="🏀", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #17408B 0%, #1d4f9f 35%, #c9803a 100%); }
    h1, h2, h3, .stMarkdown, p, label, span, div { color: #ffffff; }
    h1 { text-align: center; text-shadow: 2px 2px 0 #C9082A; letter-spacing: 1px; }
    [data-testid="stExpander"] { background: rgba(255,255,255,0.12); border: 2px solid #F58426; border-radius: 14px; }
    [data-testid="stExpander"] summary { font-weight: 700; }
    [data-testid="stMetricValue"] { color: #FFD27A; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
GIORNI_STORICO = 120
FILE_STORICO = "storico.json"


# ---------- Archivio su GitHub ----------
def gh_config():
    try:
        return st.secrets["GITHUB_TOKEN"], st.secrets["GITHUB_REPO"]
    except Exception:
        return None, None


def gh_leggi():
    token, repo = gh_config()
    if not token:
        return {}, None
    r = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{FILE_STORICO}",
        headers={"Authorization": f"Bearer {token}"}, timeout=20,
    )
    if r.status_code == 404:
        return {}, None
    r.raise_for_status()
    j = r.json()
    contenuto = base64.b64decode(j["content"]).decode("utf-8")
    return json.loads(contenuto), j["sha"]


def gh_scrivi(dati, sha):
    token, repo = gh_config()
    if not token:
        return False
    corpo = {
        "message": "Aggiorna storico pronostici",
        "content": base64.b64encode(json.dumps(dati, ensure_ascii=False, indent=1).encode("utf-8")).decode("ascii"),
    }
    if sha:
        corpo["sha"] = sha
    r = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{FILE_STORICO}",
        headers={"Authorization": f"Bearer {token}"}, json=corpo, timeout=20,
    )
    return r.status_code in (200, 201)


# ---------- Dati ESPN ----------
def chiama(data_str):
    r = requests.get(BASE + "/scoreboard", params={"dates": data_str, "limit": 100}, timeout=20)
    r.raise_for_status()
    return r.json().get("events", [])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def storico_partite(fine_str, giorni):
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
    prec = t[(t["squadra"] == squadra) & (t["data"] < giorno)]
    r = prec.tail(10)
    if len(r) < 3:
        return None
    return {
        "forma": r["vinta"].mean(),
        "fatti": r["fatti"].mean(),
        "subiti": r["subiti"].mean(),
        "diff": r["diff"].mean(),
        "riposo": min((giorno - prec["data"].max()).days, 5),
    }


def vettore(a, b):
    return [a["forma"] - b["forma"], a["fatti"] - b["fatti"], a["diff"] - b["diff"], a["riposo"] - b["riposo"]]


@st.cache_resource(show_spinner=False)
def addestra(fine_str, giorni):
    df = storico_partite(fine_str, giorni)
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
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def leader_squadra(team_id):
    risultato = {"points": [], "rebounds": [], "assists": []}
    try:
        r = requests.get(f"{BASE}/teams/{team_id}/leaders", timeout=20)
        r.raise_for_status()
        dati = r.json()
        categorie = dati.get("leaders", {}).get("categories", []) or dati.get("categories", [])
        for cat in categorie:
            nome = cat.get("name", "")
            chiave = None
            if "pointsPerGame" in nome or nome == "points":
                chiave = "points"
            elif "reboundsPerGame" in nome or nome == "rebounds":
                chiave = "rebounds"
            elif "assistsPerGame" in nome or nome == "assists":
                chiave = "assists"
            if chiave:
                for x in cat.get("leaders", [])[:3]:
                    risultato[chiave].append((
                        x.get("athlete", {}).get("displayName", "n.d."),
                        x.get("displayValue", ""),
                    ))
    except Exception:
        pass
    return risultato


def leader_dalla_partita(competitor):
    risultato = {"points": [], "rebounds": [], "assists": []}
    mappa = {"points": "points", "pointsPerGame": "points", "rebounds": "rebounds",
             "reboundsPerGame": "rebounds", "assists": "assists", "assistsPerGame": "assists"}
    for cat in competitor.get("leaders", []) or []:
        chiave = mappa.get(cat.get("name", ""))
        if chiave:
            for x in cat.get("leaders", [])[:3]:
                risultato[chiave].append((
                    x.get("athlete", {}).get("displayName", "n.d."),
                    x.get("displayValue", ""),
                ))
    return risultato


def numero(testo):
    try:
        return float(str(testo).split()[0].replace(",", "."))
    except Exception:
        return None


def mostra_leader(nome_sq, lead):
    st.markdown(f"**{nome_sq}**")
    for chiave, etichetta in {"points": "Punti", "rebounds": "Rimbalzi", "assists": "Assist"}.items():
        elenco = lead.get(chiave, [])
        if elenco:
            st.write(f"{etichetta}: " + "  |  ".join(f"{n} ({v})" for n, v in elenco))
        else:
            st.write(f"{etichetta}: dati non disponibili")


# ---------- Risultati reali di una partita ----------
@st.cache_data(ttl=1800, show_spinner=False)
def risultato_partita(event_id):
    """Restituisce punteggi finali e punti dei giocatori, se la partita è finita."""
    try:
        r = requests.get(f"{BASE}/summary", params={"event": event_id}, timeout=20)
        r.raise_for_status()
        j = r.json()
        comp = j["header"]["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            return None
        punteggi = {}
        for c in comp["competitors"]:
            punteggi[c["team"]["displayName"]] = float(c["score"])
        giocatori = {}
        for blocco in j.get("boxscore", {}).get("players", []):
            for gruppo in blocco.get("statistics", []):
                etichette = gruppo.get("labels", [])
                if "PTS" not in etichette:
                    continue
                i = etichette.index("PTS")
                for a in gruppo.get("athletes", []):
                    stats = a.get("stats", [])
                    if len(stats) > i:
                        try:
                            giocatori[a["athlete"]["displayName"]] = float(stats[i])
                        except Exception:
                            pass
        return {"punteggi": punteggi, "giocatori": giocatori}
    except Exception:
        return None


# ---------- Interfaccia ----------
st.title("🏀 Pronostici NBA")
st.caption("Stime statistiche indicative. Non garantiscono vincite. Controlla sempre gli infortuni.")

sezione = st.radio("Sezione", ["Partite di oggi", "Storico risultati"], horizontal=True)

oggi = datetime.now(ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

if sezione == "Partite di oggi":
    data_scelta = st.date_input("Giorno", value=oggi.date())
    giorno = datetime(data_scelta.year, data_scelta.month, data_scelta.day)
    giorno_str = giorno.strftime("%Y%m%d")

    try:
        with st.spinner("Carico calendario e statistiche (la prima volta ci vuole un po')..."):
            eventi = chiama(giorno_str)
            df_storico = storico_partite(giorno_str, GIORNI_STORICO)
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
    st.subheader(f"🏀 {len(eventi)} partite")
    nuovi = {}

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
        pc = (a["fatti"] + b["subiti"]) / 2
        pt = (b["fatti"] + a["subiti"]) / 2

        lead_c = leader_squadra(c["team"]["id"]) if True else {}
        if not any(lead_c.values()):
            lead_c = leader_dalla_partita(c)
        lead_t = leader_squadra(t_["team"]["id"])
        if not any(lead_t.values()):
            lead_t = leader_dalla_partita(t_)

        previsti_giocatori = {}
        for lead in (lead_c, lead_t):
            for nome, val in lead.get("points", []):
                n = numero(val)
                if n is not None:
                    previsti_giocatori[nome] = n

        nuovi[str(e["id"])] = {
            "data": giorno.strftime("%Y-%m-%d"),
            "casa": casa, "trasf": trasf,
            "favorita": fav, "prob": round(p_fav, 3),
            "punti_casa": round(pc, 1), "punti_trasf": round(pt, 1),
            "giocatori": previsti_giocatori,
        }

        with st.expander(f"🏀 {trasf} @ {casa}  |  {stato_gara}  |  Pronostico: {fav} {p_fav * 100:.0f}%"):
            k1, k2 = st.columns(2)
            k1.metric(casa + " (casa)", f"{p_casa * 100:.1f}%")
            k2.metric(trasf + " (trasferta)", f"{(1 - p_casa) * 100:.1f}%")
            st.write(f"Quota equa {casa}: {1 / p_casa:.2f}  |  {trasf}: {1 / (1 - p_casa):.2f}")
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
            else:
                st.caption("Quote non disponibili per questa partita.")

            st.markdown("### 🔥 Giocatori da seguire")
            mostra_leader(casa, lead_c)
            mostra_leader(trasf, lead_t)

            st.markdown("**Forma recente (ultime 10)**")
            st.write(f"{casa}: {a['forma'] * 10:.0f} vittorie su 10, riposo {a['riposo']} giorni")
            st.write(f"{trasf}: {b['forma'] * 10:.0f} vittorie su 10, riposo {b['riposo']} giorni")

            if p_fav >= 0.65:
                st.success("Favorita netta secondo il modello.")
            elif p_fav >= 0.55:
                st.info("Leggera favorita.")
            else:
                st.warning("Partita equilibrata, da evitare secondo il modello.")

    # salva i pronostici nuovi nello storico (solo partite non ancora iniziate o già viste)
    if nuovi and gh_config()[0]:
        try:
            archivio, sha = gh_leggi()
            cambiato = False
            for k, v in nuovi.items():
                if k not in archivio:
                    archivio[k] = v
                    cambiato = True
            if cambiato:
                if gh_scrivi(archivio, sha):
                    st.caption("Pronostici salvati nello storico.")
                else:
                    st.caption("Non sono riuscito a salvare lo storico: controlla la chiave GitHub.")
        except Exception:
            st.caption("Non sono riuscito a salvare lo storico: controlla la chiave GitHub.")
    elif nuovi:
        st.caption("Storico non attivo: manca la chiave GitHub nelle impostazioni segrete.")

else:
    st.subheader("📊 Storico: pronostico contro risultato")
    if not gh_config()[0]:
        st.warning("Storico non attivo: manca la chiave GitHub nelle impostazioni segrete di Streamlit.")
        st.stop()
    try:
        archivio, sha = gh_leggi()
    except Exception:
        st.error("Non riesco a leggere lo storico. Controlla la chiave GitHub.")
        st.stop()
    if not archivio:
        st.info("Lo storico è ancora vuoto. Apri 'Partite di oggi' per salvare i primi pronostici, poi torna qui dopo le partite.")
        st.stop()

    cambiato = False
    righe = []
    for k, v in sorted(archivio.items(), key=lambda kv: kv[1]["data"], reverse=True):
        if "risultato" not in v:
            ris = risultato_partita(k)
            if ris:
                v["risultato"] = ris
                cambiato = True
        righe.append((k, v))
    if cambiato:
        gh_scrivi(archivio, sha)

    chiuse = [(k, v) for k, v in righe if "risultato" in v]
    azzeccati = 0
    for k, v in chiuse:
        p = v["risultato"]["punteggi"]
        if v["casa"] in p and v["trasf"] in p:
            vincitore = v["casa"] if p[v["casa"]] > p[v["trasf"]] else v["trasf"]
            azzeccati += int(vincitore == v["favorita"])
    if chiuse:
        pct = azzeccati / len(chiuse) * 100
        m1, m2 = st.columns(2)
        m1.metric("Pronostici azzeccati", f"{azzeccati} su {len(chiuse)}")
        m2.metric("Percentuale", f"{pct:.0f}%")
    else:
        st.info("Nessuna partita ancora conclusa tra quelle salvate.")

    for k, v in righe:
        titolo = f"{v['data']}  |  {v['trasf']} @ {v['casa']}"
        if "risultato" not in v:
            with st.expander(f"⏳ {titolo}  |  in attesa"):
                st.write(f"Pronostico: {v['favorita']} {v['prob'] * 100:.0f}%")
                st.write(f"Punti totali stimati: {v['punti_casa'] + v['punti_trasf']:.0f}")
            continue
        p = v["risultato"]["punteggi"]
        if v["casa"] not in p or v["trasf"] not in p:
            continue
        vincitore = v["casa"] if p[v["casa"]] > p[v["trasf"]] else v["trasf"]
        ok = vincitore == v["favorita"]
        icona = "✅" if ok else "❌"
        with st.expander(f"{icona} {titolo}  |  {int(p[v['casa']])}-{int(p[v['trasf']])}"):
            st.write(f"Pronostico: {v['favorita']} ({v['prob'] * 100:.0f}%)  |  Vincitore reale: {vincitore}")
            tot_prev = v["punti_casa"] + v["punti_trasf"]
            tot_reale = p[v["casa"]] + p[v["trasf"]]
            st.write(f"Punti totali: stimati {tot_prev:.0f}  |  reali {tot_reale:.0f}  |  differenza {tot_reale - tot_prev:+.0f}")
            gioc = v["risultato"].get("giocatori", {})
            if v.get("giocatori"):
                st.markdown("**Giocatori: punti previsti contro reali**")
                for nome, prev in v["giocatori"].items():
                    if nome in gioc:
                        st.write(f"{nome}: previsti {prev:.1f}  |  reali {gioc[nome]:.0f}  |  differenza {gioc[nome] - prev:+.1f}")
                    else:
                        st.write(f"{nome}: previsti {prev:.1f}  |  non ha giocato")
st.divider()
if st.toggle("⚽ Apri la sezione Calcio"):
    from calcio import mostra_calcio
    mostra_calcio()
