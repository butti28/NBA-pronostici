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

GIORNI_STORICO = 200
ULTIME_PARTITE = 6


def chiama(lega, data_str):
    r = requests.get(f"{BASE}/{lega}/scoreboard", params={"dates": data_str, "limit": 100}, timeout=20)
    r.raise_for_status()
    return r.json().get("events", [])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def partite_concluse(lega, fine_str, giorni):
    fine = datetime.strptime(fine_str, "%Y%m%d")
    inizio = fine - timedelta(days=giorni)
    intervallo = f"{inizio.strftime('%Y%m%d')}-{(fine - timedelta(days=1)).strftime('%Y%m%d')}"
    try:
        r = requests.get(f"{BASE}/{lega}/scoreboard", params={"dates": intervallo, "limit": 1000}, timeout=40)
        r.raise_for_status()
        eventi = r.json().get("events", [])
    except Exception:
        return []
    righe = []
    for e in eventi:
        comp = e["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            continue
        sq = comp["competitors"]
        casa = next((c for c in sq if c["homeAway"] == "home"), None)
        trasf = next((c for c in sq if c["homeAway"] == "away"), None)
        if not casa or not trasf:
            continue
        righe.append({
            "id": str(e["id"]),
            "data": e.get("date", ""),
            "casa": casa["team"]["displayName"],
            "trasf": trasf["team"]["displayName"],
        })
    righe.sort(key=lambda r: r["data"])
    return righe


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def dettaglio_partita(lega, event_id):
    """Tiri in porta per squadra e per giocatore in una partita conclusa."""
    vuoto = {"squadre": {}, "giocatori": {}}
    try:
        r = requests.get(f"{BASE}/{lega}/summary", params={"event": event_id}, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return vuoto

    squadre = {}
    for sq in j.get("boxscore", {}).get("teams", []) or []:
        nome = sq.get("team", {}).get("displayName")
        for s in sq.get("statistics", []) or []:
            if s.get("name") == "shotsOnTarget":
                try:
                    squadre[nome] = float(s.get("displayValue"))
                except Exception:
                    pass

    giocatori = {}
    for ro in j.get("rosters", []) or []:
        nome_sq = ro.get("team", {}).get("displayName")
        righe = []
        for g in ro.get("roster", []) or []:
            nome = g.get("athlete", {}).get("displayName")
            if not nome:
                continue
            stats = {s.get("name"): s.get("displayValue") for s in (g.get("stats", []) or [])}
            app = stats.get("appearances")
            ha_giocato = bool(g.get("starter")) or bool(g.get("subbedIn")) or (app not in (None, "", "0", "0.0"))
            if ha_giocato:
                righe.append((nome, stats))
        trovato = any("shotsOnTarget" in stt for _, stt in righe)
        if not trovato or not nome_sq:
            continue
        elenco = {}
        for nome, stt in righe:
            try:
                elenco[nome] = float(stt.get("shotsOnTarget", 0) or 0)
            except Exception:
                elenco[nome] = 0.0
        giocatori[nome_sq] = elenco
        if nome_sq not in squadre:
            squadre[nome_sq] = sum(elenco.values())

    return {"squadre": squadre, "giocatori": giocatori}


def ultime_partite(partite, squadra):
    return [p for p in partite if squadra in (p["casa"], p["trasf"])][-ULTIME_PARTITE:]


def stima_squadra(lega, partite, squadra):
    fatti = []
    subiti = []
    for p in ultime_partite(partite, squadra):
        t = dettaglio_partita(lega, p["id"])["squadre"]
        altra = p["trasf"] if p["casa"] == squadra else p["casa"]
        if squadra in t and altra in t:
            fatti.append(t[squadra])
            subiti.append(t[altra])
    if len(fatti) < 2:
        return None
    return {
        "fatti": sum(fatti) / len(fatti),
        "subiti": sum(subiti) / len(subiti),
        "n": len(fatti),
    }


def giocatori_squadra(lega, partite, squadra):
    """I tre giocatori con più tiri in porta di media nelle ultime partite."""
    tot = {}
    for p in ultime_partite(partite, squadra):
        elenco = dettaglio_partita(lega, p["id"])["giocatori"].get(squadra, {})
        for nome, sot in elenco.items():
            t = tot.setdefault(nome, [0.0, 0])
            t[0] += sot
            t[1] += 1
    risultato = []
    for nome, valori in tot.items():
        somma, n = valori
        if n >= 2:
            risultato.append((nome, somma / n, n))
    risultato.sort(key=lambda x: x[1], reverse=True)
    return risultato[:3]


def prob_over(media, soglia):
    """Probabilità di fare almeno soglia + 1 tiri (soglia 3 significa over 3.5)."""
    cum = 0.0
    for k in range(0, soglia + 1):
        cum += math.exp(-media) * media ** k / math.factorial(k)
    return 1 - cum


def mostra_calcio():
    st.subheader("🎯 Pronostici tiri in porta")
    st.caption("Stime basate sugli ultimi incontri. Sono indicative e non garantiscono vincite.")

    nome_lega = st.selectbox("Campionato", list(CAMPIONATI.keys()))
    lega = CAMPIONATI[nome_lega]
    oggi = datetime.now(ZoneInfo("Europe/Rome")).date()
    data_scelta = st.date_input("Giorno", value=oggi, key="giorno_calcio")
    giorno_str = data_scelta.strftime("%Y%m%d")

    try:
        with st.spinner("Carico il calendario..."):
            eventi = chiama(lega, giorno_str)
            partite = partite_concluse(lega, giorno_str, GIORNI_STORICO)
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
        casa = c["team"]["displayName"]
        trasf = t_["team"]["displayName"]
        stato_gara = comp["status"]["type"].get("shortDetail", "")

        with st.expander(f"🎯 {casa} - {trasf}  |  {stato_gara}"):
            if not st.toggle("Calcola i tiri in porta", key=f"sot_{e['id']}"):
                st.caption("Accendi l'interruttore per calcolare: la prima volta ci mette qualche secondo.")
                continue

            with st.spinner("Calcolo..."):
                a = stima_squadra(lega, partite, casa)
                b = stima_squadra(lega, partite, trasf)

            if not a or not b:
                st.write("Dati sui tiri in porta non disponibili per una delle due squadre in questa competizione.")
                continue

            att_c = (a["fatti"] + b["subiti"]) / 2
            att_t = (b["fatti"] + a["subiti"]) / 2

            k1, k2, k3 = st.columns(3)
            k1.metric(casa, f"{att_c:.1f}")
            k2.metric(trasf, f"{att_t:.1f}")
            k3.metric("Totale", f"{att_c + att_t:.1f}")
            st.caption("Tiri in porta attesi.")

            st.markdown("**Squadre: probabilità di superare la soglia (stima)**")
            for nome, media in [(casa, att_c), (trasf, att_t)]:
                st.write(
                    f"{nome}: oltre 2.5 = {prob_over(media, 2) * 100:.0f}%  |  "
                    f"oltre 3.5 = {prob_over(media, 3) * 100:.0f}%  |  "
                    f"oltre 4.5 = {prob_over(media, 4) * 100:.0f}%"
                )
            tot = att_c + att_t
            st.write(
                f"Totale partita: oltre 6.5 = {prob_over(tot, 6) * 100:.0f}%  |  "
                f"oltre 8.5 = {prob_over(tot, 8) * 100:.0f}%"
            )

            st.markdown("### 🔥 Giocatori: tiri in porta")
            for nome_sq, dati_sq, attesi in [(casa, a, att_c), (trasf, b, att_t)]:
                st.markdown(f"**{nome_sq}**")
                gioc = giocatori_squadra(lega, partite, nome_sq)
                if not gioc:
                    st.write("Dati sui giocatori non disponibili.")
                    continue
                fattore = attesi / dati_sq["fatti"] if dati_sq["fatti"] > 0 else 1.0
                fattore = min(max(fattore, 0.6), 1.5)
                for nome, media, n in gioc:
                    m = media * fattore
                    p1 = 1 - math.exp(-m)
                    st.write(
                        f"{nome}: {m:.1f} attesi  |  almeno 1 tiro in porta {p1 * 100:.0f}%  |  "
                        f"media {media:.1f} in {n} partite"
                    )

            st.caption("Non considera formazioni, infortuni, rigoristi e squalifiche. Controlla sempre prima di giocare.")