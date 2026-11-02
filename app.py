"""Application Streamlit — Étude de cas 1 : lecture automatique des chèques (Jalons 1 à 4).

Placer ce fichier dans le dossier « jalon » : les modèles sont retrouvés automatiquement,
sans renommer les fichiers.   Lancer :  streamlit run app.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

BASE = Path(__file__).resolve().parent
L_MAX, VIDE = 5, 10                 # 5 positions ; 10 = case vide
PAD, DEB, FIN = 10, 11, 12          # jetons du Jalon 4
L_SEQ = L_MAX + 1
SEUIL_ALTERATION_J2 = 0.187         # seuil à 5 % de fausses alertes (notebook Jalon 2)
SEUIL_ALTERATION_J3 = 0.5           # aucun seuil fixé dans le notebook du Jalon 3

# ----------------------------------------------------------------------------------------------
# Fichiers (retrouvés par motif, même avec des noms du type « cnn_retenu (1).keras »)
# ----------------------------------------------------------------------------------------------
FICHIERS = {
    "J1 · MLP final":               "**/mlp_final_*.keras",
    "J1 · MLP référence":           "**/mlp_reference_*.keras",
    "J2 · CNN retenu (5x5)":        "**/cnn_retenu*.keras",
    "J2 · détecteur d'altération":  "**/detecteur_alteration*.keras",
    "J2 · auto-encodeur":           "**/autoencodeur_debruiteur*.keras",
    "J3 · multitâche A":            "**/multitache_A.keras",
    "J3 · multitâche B (graine 0)": "**/multitache_B_final.keras",
    "J3 · multitâche B (graine 1)": "**/multitache_B_graine1.keras",
    "J3 · multitâche B (graine 2)": "**/multitache_B_graine2.keras",
    "J4 · GRU 128 (retenu)":        "**/GRU_128_lr_0.003.weights.h5",
    "J4 · LSTM 94":                 "**/LSTM_94_lr_0.003.weights.h5",
    "J4 · positions":               "**/positions.weights.h5",
    "J4 · résultats du test":       "**/resultats_test.json",
    "Exemples d'images":            "**/exemples.npz",
}

def trouver(nom):
    trouves = sorted(BASE.glob(FICHIERS[nom]))
    return trouves[-1] if trouves else None          # le plus récent (horodatage dans le nom)

# ----------------------------------------------------------------------------------------------
# Couche personnalisée du Jalon 3 (doit être déclarée AVANT de charger les modèles)
# ----------------------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package="uc1")
class AjoutPosition(layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(shape=tuple(input_shape[1:]), initializer="random_normal", name="pos")

    def call(self, x):
        return x + self.pos

# ----------------------------------------------------------------------------------------------
# Jalon 4 : architecture reconstruite à l'identique du notebook (seuls les poids sont sauvegardés)
# ----------------------------------------------------------------------------------------------
def tronc_cnn(image):
    x = image
    for filtres, pool in ((32, (2, 2)), (64, (2, 2)), (64, (2, 1))):
        x = layers.Conv2D(filtres, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool)(x)
    return x

def construire_seq2seq(cellule="gru", unites=128):
    image = keras.Input((32, 128, 1), name="image")
    carte = tronc_cnn(image)
    h, w, c = carte.shape[1:]
    colonnes = layers.Reshape((w, h * c))(layers.Permute((2, 1, 3))(carte))
    etats = layers.Bidirectional(layers.GRU(128, return_sequences=True))(colonnes)
    resume = layers.Concatenate()([etats[:, -1, :128], etats[:, 0, 128:]])
    n_etats = 1 if cellule == "gru" else 2
    inits = [layers.Dense(unites, activation="tanh")(resume) for _ in range(n_etats)]
    encodeur = keras.Model(image, inits, name="encodeur")

    jetons = keras.Input((L_SEQ,), dtype="int32", name="jetons")
    etats_in = [keras.Input((unites,)) for _ in range(n_etats)]
    emb = layers.Embedding(13, 32)(jetons)
    rnn = (layers.GRU if cellule == "gru" else layers.LSTM)(unites, return_sequences=True)
    y = rnn(emb, initial_state=etats_in)
    probs = layers.Dense(13, activation="softmax")(y)
    decodeur = keras.Model([jetons, *etats_in], probs, name="decodeur")

    sortie_enc = encodeur(image)
    if not isinstance(sortie_enc, (list, tuple)):
        sortie_enc = [sortie_enc]
    modele = keras.Model([image, jetons], decodeur([jetons, *sortie_enc]), name=f"seq2seq_{cellule}")
    return modele, encodeur, decodeur

def construire_positions():
    image = keras.Input((32, 128, 1), name="image")
    x = layers.Flatten()(tronc_cnn(image))
    x = layers.Dense(128, activation="relu")(x)
    sorties = {f"pos{k}": layers.Dense(11, activation="softmax", name=f"pos{k}")(x) for k in range(L_MAX)}
    return keras.Model(image, sorties, name="positions")

# ----------------------------------------------------------------------------------------------
# Chargement (mis en cache)
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def charger_keras(chemin):
    return keras.models.load_model(chemin, compile=False, custom_objects={"AjoutPosition": AjoutPosition})

@st.cache_resource(show_spinner=False)
def charger_seq2seq(chemin, cellule, unites):
    modele, enc, dec = construire_seq2seq(cellule, unites)
    modele.load_weights(chemin)                      # enc et dec partagent les couches du modèle
    return enc, dec

@st.cache_resource(show_spinner=False)
def charger_positions(chemin):
    m = construire_positions()
    m.load_weights(chemin)
    return m

def modele(nom):
    chemin = trouver(nom)
    return charger_keras(str(chemin)) if chemin else None

# ----------------------------------------------------------------------------------------------
# Outils de lecture
# ----------------------------------------------------------------------------------------------
def sortie_dict(m, x):
    P = m(x, training=False)
    if isinstance(P, dict):
        return {k: np.asarray(v) for k, v in P.items()}
    noms = list(m.output.keys()) if isinstance(m.output, dict) else list(m.output_names)
    if not isinstance(P, (list, tuple)):
        P = [P]
    return {n: np.asarray(p) for n, p in zip(noms, P)}

def decoder_tetes(P, cle, ordre):
    """5 têtes softmax -> montants lus + confiance (produit des maxima).
    ordre='unites' : pos0 = unités (Jalons 1 et 4) ; ordre='gauche' : pos0 = chiffre de gauche (Jalons 2 et 3)."""
    probs = np.stack([P[cle(k)] for k in range(L_MAX)], axis=1)
    pred, mx = probs.argmax(-1), probs.max(-1)
    chiffres = lambda ligne: "".join(str(d) for d in ligne if d != VIDE)
    lus = [chiffres(l[::-1] if ordre == "unites" else l) for l in pred]
    return lus, mx.prod(-1)

def normaliser(X):
    """Normalisation par image du Jalon 3 : fond (médiane) -> 0, encre (percentile 99,5) -> 1."""
    X = np.asarray(X, "float32")
    flat = X.reshape(len(X), -1)
    bg = np.median(flat, axis=1)[:, None, None, None]
    hi = np.percentile(flat, 99.5, axis=1)[:, None, None, None]
    return np.clip((X - bg) / np.maximum(hi - bg, 1e-3), 0, 1).astype("float32")

def decoder_glouton(enc, dec, xb):
    inits = enc(xb, training=False)
    if not isinstance(inits, (list, tuple)):
        inits = [inits]
    n = len(xb)
    jetons = np.full((n, L_SEQ), PAD, "int32")
    jetons[:, 0] = DEB
    choisis = np.full((n, L_SEQ), PAD, "int64")
    pmax = np.zeros((n, L_SEQ))
    fini = np.zeros(n, bool)
    for t in range(L_SEQ):
        P = np.asarray(dec([tf.constant(jetons), *inits], training=False))[:, t]
        choisis[:, t], pmax[:, t] = P.argmax(-1), P.max(-1)
        if t + 1 < L_SEQ:
            jetons[:, t + 1] = choisis[:, t]
        fini |= choisis[:, t] >= 10
        if fini.all():
            break
    lus, confs = [], []
    for j in range(n):
        s, c = "", 1.0
        for t in range(L_SEQ):
            c *= pmax[j, t]
            if choisis[j, t] >= 10:
                break
            s += str(choisis[j, t])
        lus.append(s)
        confs.append(c)
    return lus, np.array(confs)

# ----------------------------------------------------------------------------------------------
# Un lecteur par jalon : x = image (1, 32, 128, 1) dans [0, 1]
# ----------------------------------------------------------------------------------------------
def seuil_json(nom_modele, cle):
    chemin = trouver(nom_modele)
    if chemin is None or not chemin.with_suffix(".json").exists():
        return None
    meta = json.loads(chemin.with_suffix(".json").read_text())
    return meta.get("metriques", {}).get(cle)

def lire_j1(x, nom):
    m = modele(nom)
    if m is None:
        return None
    P = sortie_dict(m, x[..., 0])                          # le MLP attend (N, 32, 128)
    lus, conf = decoder_tetes(P, lambda k: f"pos{k}", "unites")
    return {"lu": lus[0], "conf": float(conf[0]), "seuil": seuil_json(nom, "seuil_val")}

def lire_j2(x, scanner, debruiter):
    cnn = modele("J2 · CNN retenu (5x5)")
    if cnn is None:
        return None
    ae, det = modele("J2 · auto-encodeur"), modele("J2 · détecteur d'altération")
    xin, info = x, "image brute"
    if scanner == "B" and debruiter and ae is not None:
        xin, info = np.asarray(ae(x, training=False)), "débruitée par l'auto-encodeur"
    lus, conf = decoder_tetes(sortie_dict(cnn, xin), lambda k: f"pos_{k}", "gauche")
    alt = float(np.asarray(det(xin, training=False))[0, 0]) if det is not None else None
    return {"lu": lus[0], "conf": float(conf[0]), "seuil": None, "alt": alt,
            "seuil_alt": SEUIL_ALTERATION_J2, "info": info}

def lire_j3(x, scanner):
    noms = (["J3 · multitâche A"] if scanner == "A" else
            ["J3 · multitâche B (graine 0)", "J3 · multitâche B (graine 1)", "J3 · multitâche B (graine 2)"])
    mods = [m for m in (modele(n) for n in noms) if m is not None]
    if not mods:
        return None
    xn = normaliser(x)
    Ps = [sortie_dict(m, xn) for m in mods]
    P = {k: np.mean([p[k] for p in Ps], axis=0) for k in Ps[0]}
    lus, conf = decoder_tetes(P, lambda k: f"pos{k}", "gauche")
    alt = float(P["altere"][0, 0]) if "altere" in P else None
    info = f"normalisée · ensemble de {len(mods)} modèle(s)" if len(mods) > 1 else "normalisée · modèle A"
    return {"lu": lus[0], "conf": float(conf[0]), "seuil": None, "alt": alt,
            "seuil_alt": SEUIL_ALTERATION_J3, "info": info}

def seuil_j4():
    chemin = trouver("J4 · résultats du test")
    return json.loads(chemin.read_text()).get("seuil_val") if chemin else None

def lire_j4(x, nom):
    chemin = trouver(nom)
    if chemin is None:
        return None
    if nom == "J4 · positions":
        P = sortie_dict(charger_positions(str(chemin)), x)
        lus, conf = decoder_tetes(P, lambda k: f"pos{k}", "unites")
    else:
        cellule, unites = ("gru", 128) if "GRU" in nom else ("lstm", 94)
        enc, dec = charger_seq2seq(str(chemin), cellule, unites)
        lus, conf = decoder_glouton(enc, dec, x)
    return {"lu": lus[0], "conf": float(conf[0]), "seuil": seuil_j4() if "GRU" in nom else None}

# ----------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------
def preparer_image(fichier, inverser):
    """Photo/scan -> tableau (32, 128) dans [0, 1], encre claire sur fond sombre."""
    im = np.asarray(Image.open(fichier).convert("L").resize((128, 32)), "float32") / 255.0
    if inverser == "auto":
        inverser = im.mean() > 0.5
    return (1.0 - im if inverser else im).astype("float32")

@st.cache_data(show_spinner=False)
def charger_exemples(chemin):
    d = np.load(chemin, allow_pickle=True)
    return {k: d[k] for k in d.files}

# ----------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------
def lire_tout(x, scanner, debruiter, seuil_manuel, nom_j1, nom_j4):
    return {
        "Jalon 1 · MLP": lire_j1(x, nom_j1),
        "Jalon 2 · CNN": lire_j2(x, scanner, debruiter),
        "Jalon 3 · multitâche": lire_j3(x, scanner),
        "Jalon 4 · séquence": lire_j4(x, nom_j4),
    }

def afficher_carte(col, titre, res, seuil_manuel, vrai=None):
    col.subheader(titre)
    if res is None:
        col.warning("Modèle introuvable (voir « Fichiers détectés »).")
        return
    col.metric("Montant lu", res["lu"] if res["lu"] else "(vide)")
    col.metric("Confiance", f"{res['conf']:.4f}")
    seuil = res["seuil"] if res["seuil"] is not None else seuil_manuel
    origine = "seuil du notebook" if res["seuil"] is not None else "seuil manuel"
    auto = res["conf"] >= seuil
    col.write((" **automatique**" if auto else " **vers un opérateur**") + f"  \n_{origine} = {seuil:.4f}_")
    if res.get("alt") is not None:
        suspect = res["alt"] >= res["seuil_alt"]
        col.write((" **altération suspectée**" if suspect else " pas d'altération détectée")
                  + f"  \n_score {res['alt']:.3f} · seuil {res['seuil_alt']}_")
    if res.get("info"):
        col.caption(res["info"])
    if vrai is not None:
        col.write(" correct" if res["lu"] == vrai else f" erreur (vrai : {vrai})")

def page_lecture(p):
    st.header("Lire un chèque avec les 4 jalons")
    ex_chemin = trouver("Exemples d'images")
    choix = ["Téléverser une image"] + (["Exemple du scanner A", "Exemple du scanner B"] if ex_chemin else [])
    source = st.radio("Image", choix, horizontal=True, index=1 if ex_chemin else 0)

    x, vrai, scanner, altere = None, None, "A", None
    if source.startswith("Exemple"):
        ex = charger_exemples(str(ex_chemin))
        scanner = source[-1]
        X, M = ex[f"{scanner}_X"], ex[f"{scanner}_montant"]
        i = st.slider("Numéro de l'exemple", 0, len(X) - 1, 0)
        img = X[i].astype("float32")
        img = img / 255.0 if img.max() > 1.5 else img
        vrai = str(M[i])
        if f"{scanner}_altere" in ex:
            altere = int(ex[f"{scanner}_altere"][i])
    else:
        scanner = st.radio("Scanner d'origine", ["A", "B"], horizontal=True)
        fichier = st.file_uploader("Image du montant (png, jpg)", type=["png", "jpg", "jpeg"])
        inv = st.selectbox("Inverser les couleurs (encre claire sur fond sombre attendue)",
                           ["auto", "oui", "non"], index=0)
        if fichier is None:
            st.info("Téléversez une image, ou ajoutez le fichier `exemples.npz` pour utiliser des exemples.")
            return
        img = preparer_image(fichier, {"auto": "auto", "oui": True, "non": False}[inv])

    st.image(img, clamp=True, width=640, caption=f"Scanner {scanner}" + (f" · montant réel : {vrai}" if vrai else "")
             + (" · chèque altéré" if altere else ""))
    x = img[None, :, :, None].astype("float32")

    with st.spinner("Lecture en cours…"):
        res = lire_tout(x, scanner, p["debruiter"], p["seuil"], p["j1"], p["j4"])
    for col, (titre, r) in zip(st.columns(4), res.items()):
        afficher_carte(col, titre, r, p["seuil"], vrai)
    st.caption("Confiance : produit des probabilités maximales (Jalons 1 à 3) ; produit jusqu'au jeton de fin (Jalon 4).")

def page_exemples(p):
    st.header("Évaluation rapide sur les exemples")
    ex_chemin = trouver("Exemples d'images")
    if not ex_chemin:
        st.info("Ajoutez `exemples.npz` (voir `make_exemples.py`).")
        return
    ex = charger_exemples(str(ex_chemin))
    scanner = st.radio("Scanner", ["A", "B"], horizontal=True)
    X, M = ex[f"{scanner}_X"], [str(m) for m in ex[f"{scanner}_montant"]]
    st.caption(f"{len(X)} images : indicatif seulement, les vrais résultats sont ceux des notebooks (page « Résultats »).")
    if st.button("Lancer l'évaluation"):
        bons = {}
        barre = st.progress(0.0)
        for i, (img, vrai) in enumerate(zip(X, M)):
            img = img.astype("float32")
            img = img / 255.0 if img.max() > 1.5 else img
            res = lire_tout(img[None, :, :, None], scanner, p["debruiter"], p["seuil"], p["j1"], p["j4"])
            for nom, r in res.items():
                bons.setdefault(nom, []).append(None if r is None else r["lu"] == vrai)
            barre.progress((i + 1) / len(X))
        tab = pd.DataFrame({"montants exacts": {k: (np.mean(v) if v[0] is not None else np.nan) for k, v in bons.items()}})
        st.dataframe(tab.style.format("{:.2f}"))
        st.bar_chart(tab)

def lire_resultats():
    """Résultats de test issus des notebooks ; lus dans les fichiers quand ils existent."""
    lignes = {
        "Jalon 1 · MLP final":   {"A exact": 0.0972, "A auto": 0.005, "AUC alt. A": None, "B exact": None},
        "Jalon 2 · CNN 5×5":     {"A exact": 0.4600, "A auto": None, "AUC alt. A": 0.9927, "B exact": 0.4223},
        "Jalon 3 · multitâche":  {"A exact": 0.614, "A auto": None, "AUC alt. A": None, "B exact": None},
        "Jalon 4 · GRU 128":     {"A exact": 0.7756, "A auto": 0.2362, "AUC alt. A": None, "B exact": None},
    }
    chemin = trouver("J4 · résultats du test")
    if chemin:
        r = json.loads(chemin.read_text())
        lignes["Jalon 4 · GRU 128"].update({"A exact": r.get("exact_test"), "A auto": r.get("auto_test_courbe")})
    csv = next(iter(sorted(BASE.glob("**/bilan_test.csv"))), None)
    if csv is not None:
        for _, l in pd.read_csv(csv).iterrows():
            e, v = str(l.iloc[0]), l.iloc[1]
            if "Lire le montant (B)" in e:
                lignes["Jalon 3 · multitâche"]["B exact"] = v
            elif "altérations (A)" in e:
                lignes["Jalon 3 · multitâche"]["AUC alt. A"] = v
    return pd.DataFrame(lignes).T

def page_resultats():
    st.header("Résultats des jalons (ensemble de test)")
    tab = lire_resultats()
    cibles = pd.DataFrame({"A exact": [0.85], "A auto": [0.40], "AUC alt. A": [0.95], "B exact": [0.70]}, index=["Cible"])
    st.dataframe(pd.concat([tab, cibles]).astype(float).style.format("{:.3f}", na_rep="—"))
    st.caption("Valeurs issues des notebooks exécutés (test utilisé une seule fois). — : non mesuré dans ce jalon.")
    st.subheader("Exactitude du montant selon le jalon (test, scanner A)")
    st.bar_chart(tab["A exact"].astype(float))

    figures = sorted(BASE.glob("**/*.png"))
    if figures:
        st.subheader("Figures des notebooks")
        nom = st.selectbox("Figure", [f.relative_to(BASE).as_posix() for f in figures])
        st.image(str(BASE / nom))

def page_fichiers():
    st.header("Fichiers détectés")
    lignes = []
    for nom in FICHIERS:
        c = trouver(nom)
        lignes.append({"élément": nom, "trouvé": "✅" if c else "❌", "fichier": c.relative_to(BASE).as_posix() if c else ""})
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    st.write(f"TensorFlow **{tf.__version__}** · Keras **{keras.__version__}**")
    st.caption("Modèles entraînés avec TensorFlow 2.21 / Keras 3.13 (Colab). Le déploiement utilise TensorFlow 2.22 (Python 3.14), testé compatible.")

def main():
    st.set_page_config(page_title="Lecture automatique des chèques", page_icon="🧾", layout="wide")
    st.title("🧾 Lecture automatique des chèques — Banque Atlas")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", ["Lire un chèque", "Évaluer sur les exemples", "Résultats", "Fichiers détectés"])
    st.sidebar.divider()
    p = {
        "j1": st.sidebar.selectbox("Modèle Jalon 1", ["J1 · MLP final", "J1 · MLP référence"]),
        "j4": st.sidebar.selectbox("Modèle Jalon 4", ["J4 · GRU 128 (retenu)", "J4 · LSTM 94", "J4 · positions"]),
        "debruiter": st.sidebar.checkbox("Scanner B : débruiter (auto-encodeur du Jalon 2)", value=True),
        "seuil": st.sidebar.slider("Seuil de confiance manuel (modèles sans seuil du notebook)", 0.0, 1.0, 0.90, 0.01),
    }
    if page == "Lire un chèque":
        page_lecture(p)
    elif page == "Évaluer sur les exemples":
        page_exemples(p)
    elif page == "Résultats":
        page_resultats()
    else:
        page_fichiers()

if __name__ == "__main__":
    main()