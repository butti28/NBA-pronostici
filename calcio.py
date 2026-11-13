import math
import requests
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

CAMPIONATI = {
    "Serie A": "ita.1",
    "Premier League": "eng.1",
    "Liga spagnola": "esp.1",
    "Bundesliga": "ger.1",
    "Ligue 1": "fra.1",
    "Champions League": "uefa.champions",
    "Europa League": "uefa.europa",
    "Coppa Italia": "ita.coppa_italia",
    "Nations League (nazionali)": "uefa.nations",
    "Qualificazioni Europee (nazionali)": "uefa.euroq",
    "Mondiali (nazionali)": "fifa.world",
    "Amichevoli (nazionali)": "fifa.friendly",
}

GIORNI_STORICO = 400


def chiama(lega, data_str):
    r = requests.get(f"{BASE}/{lega}/scoreboard", params={"dates": data_str, "limit": 100}, timeout=20)
    r.raise_for_status()
    return r.json().get("events", [])


def gol_squadra(c):
    try:
        return float(c.get("score"))
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def partite_passate(lega, fine_str, giorni):
    fine = datetime.strptime(fine_str, "%Y%m%d")
    inizio = fine - timedelta(days=giorni)
    intervallo = f"{inizio.strftime('%Y%m%d')}-{(fine - timedelta(days=1)).strftime('%Y%m%d')}"
    righe = []
    try:
        r = requests.get(f"{BASE}/{lega}/scoreboard", params={"dates": intervallo, "limit": 1000}, timeout=30)
        r.raise_for_status()
        eventi = r.json().get("events", [])
    except Exception:
        return righe
    for e in eventi:
        comp = e["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            continue
        sq = comp["competitors"]
        casa = next((c for c in sq if c["homeAway"] == "home"), None)
        trasf = next((c for c in sq if c["homeAway"] == "away"), None)
        if not casa or not trasf:
            continue
        gc, gt = gol_squadra(casa), gol_squadra(trasf)
        if gc is None or gt is None:
            continue
        righe.append({"casa": casa["team"]["displayName"], "trasf": trasf["team"]["displayName"], "gc": gc, "gt": gt})
    return righe


def forza_squadre(righe):
    """Gol segnati e subiti medi in casa e fuori, e media del campionato."""
    if not righe:
        return None
    tot_c = sum(r["gc"] for r in righe) / len(righe)
    tot_t = sum(r["gt"] for r in righe) / len(righe)
    squadre = {}
    for r in righe:
        s = squadre.setdefault(r["casa"], {"cf": 0, "cs": 0, "cn": 0, "tf": 0, "ts": 0, "tn": 0})
        s["cf"] += r["gc"]; s["cs"] += r["gt"]; s["cn"] += 1
        s = squadre.setdefault(r["trasf"], {"cf": 0, "cs": 0, "cn": 0, "tf": 0, "ts": 0, "tn": 0})
        s["tf"] += r["gt"]; s["ts"] += r["gc"]; s["tn"] += 1
    return tot_c, tot_t, squadre


def poisson(k, media):
    return math.exp(-media) * media ** k / math.factorial(k)


def previsione(forza, casa, trasf):
    if not forza:
        return None
    media_c, media_t, squadre = forza
    a, b = squadre.get(casa), squadre.get(trasf)
    if not a or not b or a["cn"] < 2 or b["tn"] < 2:
        return None
    att_c = (a["cf"] / a["cn"]) / media_c
    dif_c = (a["cs"] / a["cn"]) / media_t
    att_t = (b["tf"] / b["tn"]) / media_t
    dif_t = (b["ts"] / b["tn"]) / media_c
    lam_c = max(0.2, att_c * dif_t * media_c)
    lam_t = max(0.2, att_t * dif_c * media_t)
    p1 = px = p2 = 0.0
    esatti = []
    for i in range(0, 8):
        for j in range(0, 8):
            p = poisson(i, lam_c) * poisson(j, lam_t)
            esatti.append(((i, j), p))
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
    somma = p1 + px + p2
    esatti.sort(key=lambda x: x[1], reverse=True)
    over25 = sum(p for (i, j), p in esatti if i + j > 2) / somma
    gol_gol = (1 - poisson(0, lam_c)) * (1 - poisson(0, lam_t))
    return {
        "p1": p1 / somma, "px": px / somma, "p2": p2 / somma,
        "lam_c": lam_c, "lam_t": lam_t,
        "esatti": [(f"{i}-{j}", p / somma) for (i, j), p in esatti[:3]],
        "over25": over25, "gol_gol": gol_gol,
    }


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def marcatori_squadra(lega, team_id):
    """Migliori marcatori di stagione della squadra, se ESPN li fornisce."""
    elenco = []
    try:
        r = requests.get(f"{BASE}/{lega}/teams/{team_id}/leaders", timeout=20)
        r.raise_for_status()
        dati = r.json()
        categorie = dati.get("leaders", {}).get("categories", []) or dati.get("categories", [])
        for cat in categorie:
            if cat.get("name", "") in ("goals", "totalGoals", "goalsPerGame"):
                for x in cat.get("leaders", [])[:3]:
                    elenco.append((x.get("athlete", {}).get("displayName", "n.d."), x.get("displayValue", "")))
    except Exception:
        pass
    return elenco


def marcatori_dalla_partita(competitor):
    elenco = []
    for cat in competitor.get("leaders", []) or []:
        if cat.get("name", "") in ("goals", "totalGoals", "goalsPerGame"):
            for x in cat.get("leaders", [])[:3]:
                elenco.append((x.get("athlete", {}).get("displayName", "n.d."), x.get("displayValue", "")))
    return elenco


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cartellini_medi(lega, fine_str, giorni):
    """Stima i cartellini medi per squadra dalle ultime partite concluse."""
    fine = datetime.strptime(fine_str, "%Y%m%d")
    inizio = fine - timedelta(days=giorni)
    intervallo = f"{inizio.strftime('%Y%m%d')}-{(fine - timedelta(days=1)).strftime('%Y%m%d')}"
    totali = {}
    try:
        r = requests.get(f"{BASE}/{lega}/scoreboard", params={"dates": intervallo, "limit": 1000}, timeout=30)
        r.raise_for_status()
        eventi = r.json().get("events", [])
    except Exception:
        return {}
    for e in eventi:
        comp = e["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            continue
        for c in comp["competitors"]:
            nome = c["team"]["displayName"]
            gialli = 0
            for s in c.get("statistics", []) or []:
                if s.get("name") == "yellowCards":
                    try:
                        gialli = float(s.get("displayValue", 0))
                    except Exception:
                        gialli = 0
            t = totali.setdefault(nome, [0.0, 0])
            t[0] += gialli
            t[1] += 1
    return {n: (v[0] / v[1]) for n, v in totali.items() if v[1] >= 2}
def mostra_calcio():
    st.subheader("⚽ Pronostici Calcio")
    st.caption("Stime statistiche indicative, soprattutto per risultato esatto, marcatori e cartellini. Non garantiscono vincite.")

    nome_lega = st.selectbox("Campionato", list(CAMPIONATI.keys()))
    lega = CAMPIONATI[nome_lega]
    oggi = datetime.now(ZoneInfo("Europe/Rome")).date()
    data_scelta = st.date_input("Giorno", value=oggi, key="giorno_calcio")
    giorno_str = data_scelta.strftime("%Y%m%d")

    try:
        with st.spinner("Carico partite e statistiche..."):
            eventi = chiama(lega, giorno_str)
            righe = partite_passate(lega, giorno_str, GIORNI_STORICO)
            forza = forza_squadre(righe)
            gialli = cartellini_medi(lega, giorno_str, GIORNI_STORICO)
    except Exception as e:
        st.error("Non riesco a scaricare i dati in questo momento. Riprova tra poco.")
        st.caption(f"Dettaglio tecnico: {e}")
        return

    if not eventi:
        st.info("Nessuna partita in questa data per questo campionato.")
        return

    st.write(f"{len(eventi)} partite")
    for e in eventi:
        comp = e["competitions"][0]
        sq = comp["competitors"]
        c = next((x for x in sq if x["homeAway"] == "home"), None)
        t_ = next((x for x in sq if x["homeAway"] == "away"), None)
        if not c or not t_:
            continue
        casa, trasf = c["team"]["displayName"], t_["team"]["displayName"]
        stato_gara = comp["status"]["type"].get("shortDetail", "")
        pr = previsione(forza, casa, trasf)

        if not pr:
            with st.expander(f"⚽ {casa} - {trasf}  |  {stato_gara}"):
                st.write("Dati insufficienti per questa partita (poche partite giocate dalle squadre).")
            continue

        esiti = {"1 (vince " + casa + ")": pr["p1"], "X (pareggio)": pr["px"], "2 (vince " + trasf + ")": pr["p2"]}
        migliore = max(esiti, key=esiti.get)

        with st.expander(f"⚽ {casa} - {trasf}  |  {stato_gara}  |  Segno: {migliore.split(' ')[0]} {esiti[migliore] * 100:.0f}%"):
            k1, k2, k3 = st.columns(3)
            k1.metric("1", f"{pr['p1'] * 100:.0f}%")
            k2.metric("X", f"{pr['px'] * 100:.0f}%")
            k3.metric("2", f"{pr['p2'] * 100:.0f}%")
            st.write(f"Quote eque: 1 = {1 / pr['p1']:.2f}  |  X = {1 / pr['px']:.2f}  |  2 = {1 / pr['p2']:.2f}")

            quote = comp.get("odds")
            if quote:
                q = quote[0]
                st.markdown("**Quote del bookmaker (ESPN)**")
                if q.get("details"):
                    st.write(f"Handicap: {q['details']}")
                if q.get("overUnder"):
                    st.write(f"Over/under gol: {q['overUnder']}")

            st.markdown("**Risultati esatti più probabili**")
            for ris, p in pr["esatti"]:
                st.write(f"{ris}: {p * 100:.1f}%")
            st.caption("Anche il risultato esatto più probabile esce raramente.")

            st.write(f"Gol attesi: {casa} {pr['lam_c']:.1f}  |  {trasf} {pr['lam_t']:.1f}")
            st.write(f"Over 2.5 gol: {pr['over25'] * 100:.0f}%  |  Gol-gol (segnano entrambe): {pr['gol_gol'] * 100:.0f}%")

            st.markdown("**⭐ Marcatori da seguire**")
            for nome_sq, comp_sq in [(casa, c), (trasf, t_)]:
                elenco = marcatori_squadra(lega, comp_sq["team"]["id"])
                if not elenco:
                    elenco = marcatori_dalla_partita(comp_sq)
                if elenco:
                    st.write(f"{nome_sq}: " + "  |  ".join(f"{n} ({v})" for n, v in elenco))
                else:
                    st.write(f"{nome_sq}: dati non disponibili")

            st.markdown("**🟨 Cartellini (stima)**")
            gc, gt = gialli.get(casa), gialli.get(trasf)
            if gc is not None and gt is not None:
                st.write(f"Gialli medi: {casa} {gc:.1f}  |  {trasf} {gt:.1f}  |  Totale stimato circa {gc + gt:.1f}")
            else:
                st.write("Dati sui cartellini non disponibili per questa partita.")

            st.caption("Gioca solo se la quota del bookmaker supera la quota equa. Controlla infortuni e squalifiche.")
