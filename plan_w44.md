# Time Series — « Réentraîner n'est pas une hygiène, c'est une décision » : le serving de prévisions comme système à contrats
 
**Dimanche 25 octobre 2026** · `W = 43` (semaine ISO du jour) · **plan étiqueté W44** (la semaine qui commence demain)
**Sélection :** `W43 → 43 mod 10 = 3` → **Time Series Engineer** · axe `43 mod 8 = 3` → **index 3 = MLOps & serving**
**Niveau :** Senior / Principal · **Budget : 195 min time-boxées**
 
> **Aucune substitution d'axe, et aucune n'était nécessaire.** L'index 3 (MLOps & serving) n'a **jamais été joué en axe
> principal** : il était présent sur les blocs 3 de W35 à W43, toujours en second. Et l'actualité tombe exactement
> dessus : **Mistral a publié Timestral le 15 octobre 2026**, une famille de foundation models de séries temporelles
> **qui tourne sur CPU**, avec des latences publiées. Un modèle qui n'a plus besoin de GPU est une **nouvelle de
> serving**, pas une nouvelle de modélisation.
>
> **Le fil rouge, en une phrase.** W40 : le protocole décide du gagnant. W41 : la métrique peut n'avoir aucun sens.
> W42 : le foundation model n'a pas les fondamentaux. W43 : le contrat d'outil remplace la fiabilité du transport.
> **W44 : le service de prévision a trois contrats incompatibles — exactitude, fraîcheur, stabilité — et la question
> « à quelle cadence réentraîner ? » n'a pas une réponse, elle en a trois séparées par deux ordres de grandeur. Celui
> qui n'écrit pas laquelle il optimise en choisit une par défaut.**
>
> **Les trois résultats mesurés cette semaine, annoncés d'emblée :**
> 1. **Connaître toutes les ruptures et réentraîner exactement dessus donne une MASE de 1,3685 — pire qu'un cron
>    hebdomadaire bête à 0,9829.** 100 % de « précision de détection » et le troisième pire score du banc. **La
>    détection de dérive répond à la mauvaise question.**
> 2. **Réentraîner rarement ne supprime pas le churn : il le concentre.** Du continu au mensuel, le **saut publié est
>    multiplié par 223,6** quand le churn moyen par pas n'est divisé que par 3,2. Et à cadence continue, **67,6 % de la
>    variance des révisions publiées est du pur bruit d'estimation** (forme close validée à 1,8 % près).
> 3. **66 % de l'amélioration qu'affiche un tableau de bord de réentraînement est de la régression vers la moyenne.**
>    Placebo mesuré : **−0,2072 pour un effet réel de −0,1053**, soit **197 % de l'effet, à partir de zéro changement**.
>    Et **plus le seuil de déclenchement est prudent, plus le faux effet grossit** : 85 % à 63,7 % du parc déclenché,
>    **284 % à 1,3 %.**
 
---
 
## 1 — Veille technologique & scientifique · 20 min
 
**1. Timestral — Mistral publie une famille de foundation models TS qui tourne sur CPU (15 octobre 2026).**
→ annonce : [mistral.ai/news/timestral](https://mistral.ai/news/timestral/)
→ fiche modèle : [docs.mistral.ai/models/timestral-1-0-209m](https://docs.mistral.ai/models/timestral-1-0-209m)
→ dépôt : [github.com/mistralai/timestral](https://github.com/mistralai/timestral)
 
**Les faits publiés, et seulement eux.** Quatre tailles partageant une architecture et une recette :
**5M / 28M / 123M / 209M** (poids fp32 : **21 / 111 / 490 / 837 Mo**). Contexte **32 768 pas** (8 192 en
pré-entraînement, ×4 en post-entraînement), sortie probabiliste sur **21 quantiles**. Innovation revendiquée :
*Decoupled Horizon Forecasting*. **Latences sur CPU serveur à 8 threads**, pour une prévision à un jour depuis trois
semaines d'historique au pas de 15 min : **10 ms (5M)** et **140 ms (209M)** ; le **28M est 5× plus rapide que le 209M**
(⇒ 28 ms) ; le **123M est à 0,2 % du score du 209M pour les trois quarts de sa latence** (⇒ 105 ms). Classements :
premier sur GIFT-Eval, TIME et fev-bench parmi les zero-shot ; le **28M bat Chronos-2, environ 4× plus gros**, sur
GIFT-Eval et TIME. Déploiement on-prem / VPC / air-gapped, statut **« Public Preview »**.
 
⚠️ **Et le piège, vérifié à la source — c'est le point de la semaine.** Le badge GitHub affiche **Apache-2.0**. Le
README précise que **le code est en Apache-2.0 mais que les poids ne le sont pas** : ils sont distribués séparément sur
le Hub sous une **« Modified MIT license »** (fiche modèle : **« Modified MIT v1.0 »**), avec une restriction d'usage
(« must not use them in a manner that infringes […] any third party's rights »). **Le texte exact de cette licence est
sur `huggingface.co`, bloqué par le proxy de cette session.**
→ **Et la même structure vaut pour Chronos-2 :** [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
est **Apache-2.0** et son README dit « This project is licensed under the Apache-2.0 License » — **sans un mot sur la
licence des poids**, qui vit dans les model cards du Hub. **La dette W42 n'est donc pas levée : elle est caractérisée.
Ce n'est pas un accident de Chronos-2, c'est un patron.** Règle d'audit à graver : **la licence des poids ne se lit
jamais dans le dépôt de code. Le badge du dépôt décrit le code.** Trois familles, trois fois le même piège, et en W42
nous l'avions pris par l'autre bout (TimesFM-3 affichait, lui, une licence franchement restrictive).
 
**2. arXiv 2604.06438 — *Cost-sensitive retraining via posterior learning debt*** (Harrison Katz — v1 **7 avril 2026**, v2 3 mai 2026, stat.AP / cs.LG)
→ [arxiv.org/abs/2604.06438](https://arxiv.org/abs/2604.06438)
Le réentraînement d'un système bayésien déployé comme **décision coûteuse** fondée sur le regret prédictif. La
quantité surveillée est la **dette d'apprentissage postérieure** : la divergence de Kullback-Leibler entre une
postérieure *fantôme* mise à jour en continu et la postérieure **gelée** du modèle déployé. On réentraîne quand le
regret attendu d'attendre une période dépasse le coût de réentraînement ; la règle usuelle à deux états en est un cas
particulier. **Les chiffres :** une politique à seuil de dette ajustée par l'âge **bat le calendaire réglé dans 72
cellules de scénario non stables sur 72**, et **bat le CUSUM réglé dans 58 sur 72** ; objectifs relatifs moyens de
**0,677 contre le calendaire** et **0,975 contre le CUSUM**. Les auteurs précisent eux-mêmes qu'il s'agit d'une
**preuve de concept sur simulation conjuguée synthétique**, et non d'un remplacement universel de la détection de
dérive — honnêteté qui rend le papier utilisable.
*Pourquoi cette semaine :* c'est le formalisme exact du bloc 2, et le 0,975 contre CUSUM dit déjà quelque chose
d'important : **la dette KL ne gagne pas grand-chose contre un bon CUSUM ; ce qu'elle apporte, c'est une quantité
lisible et un seuil exprimé dans l'unité du coût.**
 
**3. arXiv 2506.05776 — *Analyzing the retraining frequency of global forecasting models: towards more stable forecasting systems*** (Marco Zanotti — v3 **10 février 2026**, dépôt initial 6 juin 2025, stat.AP / stat.OT)
→ [arxiv.org/abs/2506.05776](https://arxiv.org/abs/2506.05776)
Trois grands jeux, dix modèles globaux, plusieurs calendriers de réentraînement, et **une métrique nouvelle : le
*Scaled Multi-Quantile Change* (SMQC)**, agnostique au modèle, sans hypothèse de distribution et sans échelle, pour
mesurer la **stabilité** des prévisions probabilistes. Conclusion : **réentraîner moins souvent tend à préserver ou
améliorer la stabilité**, et exactitude et stabilité ne sont pas nécessairement en conflit en modélisation globale.
À lire avec son compagnon, [arXiv 2505.00356](https://arxiv.org/abs/2505.00356) (*On the retraining frequency of
global models in retail demand forecasting*, v3 **14 janvier 2026**, DOI 10.1016/j.mlwa.2025.100769) : dix modèles,
deux jeux de distribution, du réentraînement continu à l'absence totale de réentraînement — **un réentraînement moins
fréquent garde une exactitude à peu près stable pour un coût de calcul plus bas.**
*Pourquoi cette semaine :* parce que le SMQC nomme ce que personne ne mesure. **La stabilité d'un service de prévision
est une clause de service, pas une métrique d'exactitude** — le bloc 4 en donne la forme close, le bloc 5 la
transforme en clause signable. ⚠️ **Et les deux abstracts ne contiennent aucun chiffre** : les résultats numériques
sont dans le corps des papiers. Je les cite donc pour leur **cadre**, pas pour des valeurs que je n'ai pas vérifiées.
 
**4. arXiv 2609.39789 — *Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting*** (Kwak, Jung, Park — **30 septembre 2026**, cs.LG / cs.AI)
→ [arxiv.org/abs/2609.39789](https://arxiv.org/abs/2609.39789)
**PILOT** construit une **pseudo-étiquette** à partir des **hausses futures de l'erreur de prévision**, puis entraîne
un scoreur léger à prédire cette étiquette depuis l'état d'erreur observé. Au déploiement il n'utilise que des
**erreurs déjà closes**, et se branche sur n'importe quel backbone sans modification d'architecture. Évalué sur
**8 jeux multivariés** et **3 backbones** (DLinear, iTransformer, TimesNet) : meilleur rang moyen parmi les politiques
de réentraînement, avec un bon compromis performance/efficacité.
⚠️ **Déclassé en partie :** l'abstract **ne donne aucun autre chiffre** — ni gain, ni coût, ni écart aux baselines.
Retenu pour **une seule idée, qui est bonne** : l'étiquette d'entraînement du déclencheur est une quantité **future**,
et le déclencheur déployé ne lit que des erreurs **closes**. **C'est la seule façon correcte de construire un
déclencheur apprenant**, et c'est exactement la discipline de point-in-time de W38 appliquée au méta-niveau.
 
> **Ressource écartée, et la règle qui la fait écarter.** arXiv 2601.08928 — *DriftGuard: A Hierarchical Framework for
> Concept Drift Detection and Remediation in Supply Chain Forecasting* (Alam, Rahman, Sadeqa, 13 janvier 2026) →
> [arxiv.org/abs/2601.08928](https://arxiv.org/abs/2601.08928). Cinq modules, quatre détecteurs, SHAP pour le
> diagnostic, réentraînement conscient du coût, **plus de 30 000 séries M5**, **97,8 % de rappel de détection en
> 4,2 jours**, et **« un retour sur investissement allant jusqu'à 417 »** — **sans unité**. La règle de W43 s'applique
> telle quelle : **une revendication chiffrée sans unité n'est pas un chiffre.** Deux signaux supplémentaires : le
> papier est classé **cs.CV** pour un sujet de prévision de chaîne logistique, et M5 reste un corpus de compétition.
> **Gardé comme piste, retiré de la veille.** Et surtout : le bloc 2 montre que **97,8 % de rappel de détection est le
> mauvais KPI** — l'oracle de détection de ce banc a 100 % de rappel et perd contre un cron.
 
### Synthèse — ce qui bouge, ce que ça change pour moi
 
1. **Ce qui bouge :** le serving de prévisions passe du GPU au CPU (Timestral : 140 ms pour 209M sur 8 threads), ce qui
   déplace la contrainte du **coût de calcul** vers la **fenêtre de calcul** et la **licence des poids**. En parallèle,
   la littérature 2026 converge sur un point que la pratique ignore : **la cadence de réentraînement est une décision
   économique** (2604.06438) et **la stabilité est une dimension à part entière** (2506.05776, SMQC).
2. **Ce que ça change pour moi :** je ne vends plus « on met le modèle en production ». Je vends **le contrat de
   service de prévision** : budget de fraîcheur **indexé sur le quantile servi**, clause de stabilité chiffrée, gate
   de licence des poids, témoin permanent pour que l'effet d'un réentraînement soit mesurable — et le **calcul en
   euros** de la cadence.
3. **Ce que je refuse de croire sans l'avoir mesuré :** qu'un détecteur de dérive dise quand réentraîner. Le bloc 2 le
   teste avec un détecteur **parfait** et il perd.
> **La surcote de la semaine, et elle est partout : « on a mis en place un monitoring de drift ».**
> La dérive détectable n'est pas la dérive coûteuse. Un détecteur parfait de ruptures de niveau atteint **100 % de
> réentraînements utiles** sur mon banc et finit **3ᵉ pire en MASE sur 10 politiques** (1,3685 contre 0,9829 pour un
> cron hebdomadaire) — parce que **la dérive lente du profil saisonnier n'est pas une rupture**, et que c'est elle qui
> coûte. Corollaire commercial : quand un client dit « on détecte la dérive », la bonne question n'est pas *comment*,
> c'est **« et qu'est-ce que vous en faites, et de combien votre erreur a-t-elle baissé, mesuré contre un témoin ? »**
 
---
 
## 2 — Ingénierie & code · 35 min
 
### Objectif
Mesurer, sur un parc servi en continu, **ce que coûte et ce que rapporte chaque politique de réentraînement** — et
établir trois faits qui contredisent la pratique : (1) un détecteur de dérive parfait perd contre un cron ; (2) la
latence de clôture des données crée une **illusion de backtest** chiffrable ; (3) la cadence ne supprime pas le churn,
elle choisit sa distribution.
 
### Consigne précise
- Parc de **120 séries × 2 400 pas**, saisonnalité `M = 24`, horizon servi `H = 24`, fenêtre d'entraînement `W = 240`.
- DGP à **deux dérives de nature différente** : des **ruptures de niveau** (Poisson, brutales) **et** une dérive lente
  de l'**amplitude saisonnière** (sinusoïdale, à phase propre par série). C'est le cœur du dispositif : une politique
  qui ne réagit qu'aux ruptures est aveugle à la seconde.
- **Latence de clôture des données `d = 48`** (deux jours) : un modèle « réentraîné maintenant » est en réalité ajusté
  sur une fenêtre fermée à `t − d`. Le banc tourne en `d = 48` (la réalité) et en `d = 0` (la fuite) pour chiffrer
  l'écart.
- Dix politiques : `jamais`, calendaire `k ∈ {1, 24, 48, 168, 720}`, **CUSUM bilatéral sur l'erreur réalisée**,
  **dette KL** (deux seuils), et **`ruptures CONNUES`** — un détecteur de dérive **parfait**, qui n'est pas un oracle
  de réentraînement et c'est tout le propos.
- Trois familles de mesures, jamais une : **MASE**, **coût** (nombre de réentraînements), **churn** — et le churn se
  décline en **amplitude du saut p95** et **fréquence des sauts visibles (> 1 % de l'échelle)**.
### Stack
`python 3.11+`, **`numpy` seul**. Sans réseau, sans GPU, sans clé API : exécutable dans n'importe quel notebook client.
 
### Code — **exécuté, sorties réelles ci-dessous**
 
```python
"""Bloc 2 - W44 : reentrainer n'est pas une hygiene, c'est une decision couteuse.
 
Banc de politiques de reentrainement sur un parc de series servies en continu.
Trois questions, et aucune n'est "quel modele" :
  1. a quelle cadence reentrainer, sous un cout explicite ?
  2. combien coute la LATENCE DE DONNEES -- les actuals des d derniers pas ne
     sont pas encore clotures, donc un modele "reentraine maintenant" est en
     realite entraine sur des donnees vieilles de d pas. Un backtest qui
     l'ignore est une FUITE TEMPORELLE de serving.
  3. le CHURN : amplitude des revisions successives de la MEME date cible.
     C'est une clause de service, et il n'apparait dans aucune metrique
     d'exactitude.
 
Dependances : numpy seul.
"""
from __future__ import annotations
 
import numpy as np
 
M = 24           # saisonnalite journaliere horaire
H = 24           # horizon servi
W = 240          # fenetre d'entrainement (10 jours)
T = 2400         # longueur de serie
N = 120          # taille du parc
D_LAG = 48       # latence de cloture des donnees (2 jours) -- le coeur du bloc
WARM = W + M + D_LAG
 
 
def make_park(n=N, T=T, seed=44, rate=1 / 600.0, jump=0.8, sigma=0.5):
    """Niveau par morceaux (ruptures) + profil saisonnier qui derive lentement.
    Les ruptures sont brutales ; la derive saisonniere ne l'est pas. C'est le
    point : une politique qui ne reagit qu'aux ruptures ignore la derive."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    seas_base = rng.normal(0, 1.2, (n, M))
    seas_base -= seas_base.mean(axis=1, keepdims=True)
    cps = rng.random((n, T)) < rate
    cps[:, :WARM] = False
    level = 10.0 + np.cumsum(rng.normal(0, jump, (n, T)) * cps, axis=1)
    amp = 1.0 + 0.35 * np.sin(2 * np.pi * t / T + rng.uniform(0, 6.28, (n, 1)))
    seas = seas_base[:, t % M] * amp
    y = level + seas + rng.normal(0, sigma, (n, T))
    return dict(y=y, cps=cps, sigma=sigma)
 
 
def fit(y, t_end, w=W):
    """Ajuste sur la fenetre FERMEE a t_end inclus (t_end-w+1 .. t_end)."""
    yw = y[:, t_end - w + 1:t_end + 1]
    mu = yw.mean(axis=1)
    resid = yw - mu[:, None]
    phase = np.arange(t_end - w + 1, t_end + 1) % M
    prof = np.empty((y.shape[0], M))
    for k in range(M):
        prof[:, k] = resid[:, phase == k].mean(axis=1)
    return mu, prof
 
 
def hfc(mu, prof, t0):
    ph = np.arange(t0 + 1, t0 + H + 1) % M
    return mu[:, None] + prof[:, ph]
 
 
# --------------------------------------------------------------------------- #
# Politiques : rendent un masque (n,) "reentrainer a cette origine".
# --------------------------------------------------------------------------- #
def p_never(s, t, c):
    return np.zeros(c["n"], bool)
 
 
def p_calendar(s, t, c):
    return np.full(c["n"], t > WARM and (t - WARM) % s["k"] == 0)
 
 
def _cusum(s, t, c, err):
    ok = np.isfinite(err)
    z = np.where(ok, err / s["sigma"], 0.0)
    s["sp"] = np.maximum(0.0, s["sp"] + z - s["slack"])
    s["sm"] = np.maximum(0.0, s["sm"] - z - s["slack"])
    fire = ((s["sp"] > s["thr"]) | (s["sm"] > s["thr"])) & ok
    s["sp"] = np.where(fire, 0.0, s["sp"]); s["sm"] = np.where(fire, 0.0, s["sm"])
    return fire
 
 
def p_cusum(s, t, c):
    return _cusum(s, t, c, c["err"])
 
 
def p_debt(s, t, c):
    """Dette d'apprentissage = KL(posterieure fantome || posterieure deployee),
    gaussienne a variance connue (cf. arXiv 2604.06438, qui l'ecrit en
    normale-inverse-gamma). On reentraine quand la dette depasse un seuil
    exprime dans la meme unite que le cout de reentrainement."""
    s2 = s["sigma"] ** 2
    v_s, v_d = s2 / W, s2 / W
    kl = (c["shadow_mean"] - s["m_d"]) ** 2 / (2 * v_d) + \
         0.5 * np.log(v_d / v_s) + v_s / (2 * v_d) - 0.5
    return kl > s["thr"]
 
 
def p_cps(s, t, c):
    """Ruptures CONNUES : reentraine au pas qui suit chaque rupture.
    Ce n'est PAS un oracle de reentrainement -- c'est un oracle de DETECTION
    DE DERIVE. Le banc montre que ce ne sont pas la meme chose."""
    return s["cps"][:, t]
 
 
# --------------------------------------------------------------------------- #
def run(park, policy, st, fit_lag=D_LAG):
    """fit_lag : age des donnees utilisables. fit_lag = D_LAG est la realite ;
    fit_lag = 0 est la fuite (on pretend que les actuals sont clotures)."""
    y = park["y"]; n = y.shape[0]
    s = dict(st); s.setdefault("sigma", park["sigma"])
    mu, prof = fit(y, WARM - fit_lag)
    s["m_d"] = mu.copy(); s["sp"] = np.zeros(n); s["sm"] = np.zeros(n)
 
    F_prev = None; pend = {}
    ae = np.zeros(n); rev = []; nre = np.zeros(n); hit = np.zeros(n)
    cnt = 0
    for t in range(WARM, T - H):
        err = np.full(n, np.nan)
        if t in pend:
            err = y[:, t] - pend.pop(t)
        sw = y[:, t - fit_lag - W + 1:t - fit_lag + 1]
        c = dict(n=n, err=err, shadow_mean=sw.mean(axis=1))
 
        F = hfc(mu, prof, t)
        ae += np.abs(F - y[:, t + 1:t + H + 1]).mean(axis=1); cnt += 1
        pend[t + H] = F[:, -1].copy()
        if F_prev is not None:
            rev.append(np.abs(F[:, :-1] - F_prev[:, 1:]).mean(axis=1))
        F_prev = F
 
        fire = policy(s, t, c)
        if fire.any():
            mu_n, pr_n = fit(y, t - fit_lag)
            mu = np.where(fire, mu_n, mu)
            prof = np.where(fire[:, None], pr_n, prof)
            s["m_d"] = np.where(fire, mu_n, s["m_d"])
            nre += fire
            hit += fire & park["cps"][:, max(t - 72, 0):t + 1].any(axis=1)
 
    scale = np.abs(y[:, M:] - y[:, :-M]).mean(axis=1)
    rev = np.asarray(rev) / scale[None, :]        # revision RELATIVE (origines, series)
    nz = rev[rev > 1e-12]
    # Un modele GELE a un churn exactement nul : le churn est entierement
    # concentre sur les origines de reentrainement. La clause de service n'est
    # donc pas la moyenne, c'est l'AMPLITUDE DU SAUT quand il y en a un, et la
    # FREQUENCE des sauts visibles par le metier.
    return dict(mase=float((ae / cnt / scale).mean()),
                churn=float(rev.mean()),
                jump_p95=float(np.percentile(nz, 95)) if nz.size else 0.0,
                pct_visible=float((rev > 0.01).mean()),
                nre=float(nre.mean()),
                useful=float(hit.sum() / max(nre.sum(), 1)))
 
 
if __name__ == "__main__":
    park = make_park()
    print(f"parc : {N} series x {T} pas | horizon H = {H} | fenetre W = {W} | "
          f"latence de donnees d = {D_LAG}")
    print(f"ruptures de niveau : {park['cps'].sum(axis=1).mean():.1f} par serie "
          f"({int(park['cps'].sum())} au total)\n")
 
    POL = [("jamais",                     p_never,    {}),
           ("calendaire k=720 (mensuel)", p_calendar, {"k": 720}),
           ("calendaire k=168 (hebdo)",   p_calendar, {"k": 168}),
           ("calendaire k=48",            p_calendar, {"k": 48}),
           ("calendaire k=24 (quotidien)",p_calendar, {"k": 24}),
           ("calendaire k=1 (continu)",   p_calendar, {"k": 1}),
           ("CUSUM sur erreur realisee",  p_cusum,    {"thr": 5.0, "slack": 0.5}),
           ("dette KL (seuil 8)",         p_debt,     {"thr": 8.0}),
           ("dette KL (seuil 2)",         p_debt,     {"thr": 2.0}),
           ("ruptures CONNUES",           p_cps,      {"cps": park["cps"]})]
 
    print(f"{'politique':>29s} {'MASE':>7s} {'churn':>8s} {'saut p95':>9s} "
          f"{'% orig. >1%':>12s} {'reentr.':>8s} {'% a <72h':>9s}")
    res = {}
    for nm, f, st in POL:
        r = run(park, f, st); res[nm] = r
        print(f"{nm:>29s} {r['mase']:>7.4f} {r['churn']:>8.4f} "
              f"{r['jump_p95']:>9.4f} {r['pct_visible']:>11.1%} "
              f"{r['nre']:>8.1f} {r['useful']:>8.1%}")
 
    print("\n--- A. la latence de donnees : l'illusion de backtest ---")
    print(f"{'politique':>29s} {'MASE reelle (d=48)':>19s} {'MASE fuyante (d=0)':>19s} "
          f"{'illusion':>9s}")
    for nm, f, st in [POL[4], POL[2], POL[6], POL[8]]:
        a = run(park, f, dict(st), fit_lag=D_LAG)
        b = run(park, f, dict(st), fit_lag=0)
        print(f"{nm:>29s} {a['mase']:>19.4f} {b['mase']:>19.4f} "
              f"{(a['mase']-b['mase'])/b['mase']:>+8.1%}")
 
    print("\n--- B. le cout total : exactitude + reentrainements + churn ---")
    print("   J = MASE + lam_c * (reentr./1000) + lam_k * churn")
    print(f"{'lam_c':>7s} {'lam_k':>7s} {'politique optimale':>29s} {'J':>8s}")
    for lam_c, lam_k in ((0.0, 0.0), (0.05, 0.0), (0.05, 2.0),
                         (0.05, 10.0), (0.20, 10.0), (0.05, 40.0)):
        best = min(res.items(),
                   key=lambda kv: kv[1]["mase"] + lam_c * kv[1]["nre"] / 1000
                   + lam_k * kv[1]["churn"])
        J = best[1]["mase"] + lam_c * best[1]["nre"] / 1000 + lam_k * best[1]["churn"]
        print(f"{lam_c:>7.2f} {lam_k:>7.1f} {best[0]:>29s} {J:>8.4f}")
```
 
### Sorties obtenues
 
Parc : 120 séries × 2 400 pas, `H = 24`, `W = 240`, `d = 48`. **3,7 ruptures de niveau par série** (448 au total).
 
| politique | MASE | churn moyen | saut p95 | % origines > 1 % | réentr./série | % à < 72 h d'une rupture |
|---|---|---|---|---|---|---|
| jamais | 1,6003 | 0,0000 | — | 0,0 % | 0,0 | 0,0 % |
| calendaire k=720 (mensuel) | 1,2520 | 0,0012 | **2,9263** | 0,1 % | 2,0 | 8,8 % |
| calendaire k=168 (hebdo) | 0,9829 | 0,0028 | 1,2647 | 0,6 % | 12,0 | 11,6 % |
| calendaire k=48 | 0,9177 | 0,0039 | 0,4153 | 2,0 % | 42,0 | 12,2 % |
| calendaire k=24 (quotidien) | 0,9042 | 0,0049 | 0,2153 | 4,1 % | 85,0 | 12,2 % |
| **calendaire k=1 (continu)** | **0,8915** | 0,0052 | **0,0132** | 12,6 % | 2 063,0 | 12,2 % |
| CUSUM sur erreur réalisée | 0,9189 | 0,0031 | 0,2797 | 4,2 % | 92,5 | **40,9 %** |
| dette KL (seuil 8) | 1,0025 | 0,0020 | 0,6114 | 0,7 % | 13,9 | 13,8 % |
| dette KL (seuil 2) | 0,9546 | 0,0026 | 0,4189 | 1,5 % | 30,3 | 15,3 % |
| **ruptures CONNUES** | **1,3685** | 0,0015 | 2,2602 | 0,2 % | **3,7** | **100,0 %** |
 
**A. La latence de données : l'illusion de backtest.**
 
| politique | MASE réelle (`d = 48`) | MASE fuyante (`d = 0`) | illusion |
|---|---|---|---|
| calendaire k=24 | 0,9042 | 0,8510 | **+6,3 %** |
| calendaire k=168 | 0,9829 | 0,9304 | +5,6 % |
| CUSUM | 0,9189 | 0,8655 | +6,2 % |
| dette KL (seuil 2) | 0,9546 | 0,8983 | +6,3 % |
 
**B. Le coût total `J = MASE + λ_c · (réentr./1000) + λ_k · churn`, et la politique optimale qui en découle :**
 
| λ_c | λ_k | politique optimale | J |
|---|---|---|---|
| 0,00 | 0,0 | calendaire k=1 (continu) | 0,8915 |
| 0,05 | 0,0 | calendaire k=24 | 0,9085 |
| 0,05 | 2,0 | calendaire k=24 | 0,9183 |
| 0,05 | 10,0 | **CUSUM** | 0,9548 |
| 0,20 | 10,0 | calendaire k=48 | 0,9647 |
| 0,05 | 40,0 | **CUSUM** | 1,0488 |
 
### Les quatre lectures, et la deuxième est le résultat de la semaine
 
**1. Le détecteur de dérive parfait perd contre un cron hebdomadaire.** `ruptures CONNUES` a **100,0 % de
réentraînements « utiles »** (tous à moins de 72 h d'une rupture réelle), **3,7 réentraînements par série**, et une
**MASE de 1,3685** — contre **0,9829 pour un calendaire hebdomadaire** qui en fait 12 et dont **11,6 %** seulement
tombent près d'une rupture. **Le taux de détection et le résultat classent les politiques dans des ordres opposés.**
Mécanisme : la dérive de l'amplitude saisonnière n'est pas une rupture, donc elle n'est jamais détectée, donc elle
n'est jamais corrigée. **Troisième reproduction du même motif en quatre semaines** : W41 (le prévisionniste nul premier
en MASE), W43 (le linter v0 à 100 % de précision et 45 violations bloquantes manquées, et l'agent à 87,3 % de bons
choix pour une MASE pire), W44 (100 % de détection, 3ᵉ pire MASE). **Une métrique de processus ne vaut jamais une
métrique de résultat, et c'est devenu une règle de métier, pas une anecdote.**
 
**2. Réentraîner rarement ne supprime pas le churn : il le concentre — et personne ne mesure la bonne chose.** Un
modèle **gelé a un churn exactement nul** : le churn est **entièrement concentré sur les origines de réentraînement**.
Donc la moyenne est inutilisable. Les deux quantités qui comptent sont **l'amplitude du saut** et **sa fréquence**, et
elles vont **en sens opposés** :
- mensuel : saut p95 = **2,9263** (293 % de l'échelle MASE !) mais **0,1 %** des origines ;
- continu : saut p95 = **0,0132** (1,3 %) mais **12,6 %** des origines.
**Un facteur 222 sur l'amplitude, un facteur 126 sur la fréquence, en sens inverse.** Le churn *moyen* ne bouge que
d'un facteur 4,3 — et c'est précisément pourquoi il ne faut pas le regarder. **Deux clauses de service sont donc
mathématiquement incompatibles : « aucune révision ne dépasse X % » pousse vers le continu, « au plus N révisions par
mois » pousse vers le mensuel.** Le bloc 4 en donne la forme close et le bloc 5 la clause signable.
 
**3. La latence de clôture des données vaut +6,3 % de MASE, invisible et systématique.** Ajuster sur une fenêtre fermée
à `t` quand les actuals ne sont clos qu'à `t − 48` est une **fuite temporelle de serving** : elle ne vient pas d'une
jointure mais du **calendrier de clôture**. Elle est **du même signe et du même mécanisme** que le `+172,6 %` de W38
(`leak_premium.py`), en beaucoup plus discret — donc plus dangereuse : personne ne la cherche. **Et elle est stable
d'une politique à l'autre (+5,6 % à +6,3 %), donc elle ne change pas le classement : elle déplace toute la courbe.
C'est la raison pour laquelle on ne la voit pas.**
 
**4. La politique optimale change de famille avec le prix de la stabilité.** À λ_k = 0 c'est le continu ; à λ_k = 2 le
quotidien ; à λ_k = 10 **le CUSUM**, qui n'est premier sur aucune des trois métriques prises isolément. **Le CUSUM
gagne exactement là où il faut payer la stabilité**, parce qu'il réentraîne 92,5 fois en plaçant **40,9 %** de ses
réentraînements près d'une rupture, contre 12,2 % pour le calendaire de budget comparable — **3,4× mieux ciblé pour un
budget équivalent.** C'est la seule politique du banc dont la valeur ne se voit qu'en objectif composite.
 
### Les pièges techniques de la semaine (quatre)
1. **La latence de clôture des données.** Mesuré : **+6,3 %** d'illusion de MASE. Un modèle réentraîné « maintenant »
   n'a jamais vu les `d` derniers pas. **Le backtest doit porter la même latence que la production.**
2. **Le churn moyen d'un modèle gelé est nul par construction.** Mesurer le churn en moyenne conclut toujours que
   réentraîner moins est plus stable — ce qui est faux pour l'amplitude. **La bonne mesure est conditionnelle à
   l'événement de réentraînement.**
3. **Le dénominateur de MASE calculé sur toute la série**, ruptures incluses, gonfle l'échelle et flatte toutes les
   politiques à l'identique. Ici il est explicite (`|y_t − y_{t−24}|` à la main, **jamais `np.diff(a, n=24)`** — piège
   de W42, reconduit tel quel).
4. **Confondre « réentraîner » et « réajuster la fenêtre ».** Le bloc 4 montre en forme close que **la longueur de
   fenêtre et la cadence règlent deux termes additifs indépendants** : l'optimum de fenêtre `w* = 27` est le **même
   pour `k = 1`, `k = 24` et `k = 168`**. Les équipes resserrent le cron en croyant corriger un biais de fenêtre.
### « Tu sais que c'est fini quand… »
`ruptures CONNUES` affiche **100 % de réentraînements utiles et une MASE strictement pire que le calendaire
hebdomadaire** ; le saut p95 et le pourcentage d'origines visibles **classent les politiques en ordre inverse** ; et la
MASE en `d = 0` est **meilleure** que la MASE en `d = 48` pour **les quatre** politiques testées. Si le détecteur
parfait gagne, le DGP n'a qu'une seule nature de dérive — et le banc ne prouve rien.
 
---
 
## 3 — System Design & MLOps · 20 min
 
### Décision à trancher
Un parc de prévisions servi derrière une API, un réentraînement orchestré, un cache. **Quel contrat de service
écrire — et que faire du fait que le protocole d'inférence standard de l'industrie ne peut pas exprimer une
prévision ?**
 
### Le fait vérifié qui commande toute l'architecture
 
**L'Open Inference Protocol V2 (KServe / Seldon, l'interface de MLServer, Triton, TorchServe…) n'a aucun champ pour le
temps.** Vérifié champ par champ sur la spécification
([docs.seldon.ai/seldon-core-2/user-guide/v2](https://docs.seldon.ai/seldon-core-2/user-guide/v2), 25/10/2026) :
 
- requête : `id`, `parameters`, `inputs[]`, `outputs[]` ;
- `inputs[]` : `name`, `shape`, `datatype`, `parameters`, `data` ;
- réponse : `model_name`, `model_version`, `id`, `parameters`, `outputs[]` ;
- métadonnées de modèle : `name`, `versions`, `platform`, `inputs`, `outputs`.
**Aucun champ `as_of`, aucun horodatage, aucun `horizon`, aucun niveau de quantile.** Le seul point d'extension est
l'objet `parameters` générique — et **la spécification dit elle-même « Currently no parameters are defined »**, donc
tout ce qu'on y met est **propre au serveur et non portable**.
 
**Trois conséquences qu'il faut nommer en entretien.**
1. **Deux requêtes identiques ne sont pas la même requête** si elles sont émises à deux dates différentes. Sans
   `as_of` dans le contrat, **un cache de prévisions n'est pas spécifiable** : on ne peut pas dire ce qu'il est licite
   de resservir. W43 avait `ttlMs` / `cacheScope` rendus **obligatoires** par le MCP `2026-07-28` ; l'OIP **n'a rien
   d'équivalent**. Le protocole le plus répandu pour servir des modèles est, sur ce point, en retard sur un protocole
   d'outils d'agent de trois mois.
2. **`model_version` existe, `feature_version` et `training_asof` n'existent pas.** La traçabilité du protocole
   s'arrête au binaire du modèle. Le `pinned_together` de W43 doit donc être porté **par-dessus** le protocole.
3. **Le niveau de service est hors contrat.** Les 21 quantiles de Timestral entrent dans un tenseur `data` dont la
   sémantique est… la convention du serveur. **Le bloc 5 montre que la cadence admissible dépend d'un facteur 7 du
   quantile servi** : une information décisive, qu'aucun champ ne transporte.
```
 ┌──────────── GATE DE DEPLOIEMENT (deterministe, 0 token) ─────────────────────┐
 │ G1 licence des POIDS lue a la source et listee ?   <- bloquant (bloc 6)      │
 │ G2 latence p99 x taille de lot <= fenetre de calcul ?          <- bloquant   │
 │ G3 cadence <= k_max(quantile servi) ?   (tau=0.95 -> 24 h)     <- bloquant   │
 │ G4 saut p95 <= clause contractuelle ?                          <- bloquant   │
 │ G5 temoin permanent >= 2 % du parc configure ?                 <- bloquant   │
 │ G6 latence de cloture d declaree ET appliquee au backtest ?    <- bloquant   │
 └────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
 ┌──────────── PLAN DE DONNEES : LA BITEMPORALITE EST DANS LE SERVING ─────────┐
 │                                                                            │
 │  evenement ──(clos a t+d)──► entrepot ──(fenetre fermee a t-d)──► ENTRAIN.  │
 │                                   │                                        │
 │  REGLE : le backtest lit la MEME fenetre que la production. Mesure du prix  │
 │  de la violation : +6,3 % de MASE (bloc 2).                                │
 └────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
 ┌──────────── SERVING : le contrat vit AU-DESSUS du protocole ────────────────┐
 │  client ──► passerelle de prevision ──────► runtime OIP V2 ──► modele      │
 │               │  ajoute as_of, horizon, tau[], training_asof,              │
 │               │  model_version, feature_version   (OIP: aucun champ)        │
 │               │  calcule la cle de cache = (serie, as_of, horizon, tau)     │
 │               └─ refuse de servir si age(modele) > budget_fraicheur(tau)    │
 │                                                                            │
 │  cache : ttl INDEXE SUR L'HORIZON (h=1 : minutes ; h=24 : heures)           │
 │  bascule : bleu/vert par COHORTE de series + temoin permanent non bascule   │
 └────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
 ┌──────────── OBSERVABILITE : 4 signaux, et 3 n'existent nulle part ──────────┐
 │ 1. age du modele servi, p95 PAR SERIE   <- pas la moyenne : la queue       │
 │ 2. saut p95 de revision + % d'origines > 1 %   <- la clause de service     │
 │ 3. part de BRUIT dans la variance des revisions publiees (bloc 4)          │
 │ 4. effet du reentrainement mesure CONTRE LE TEMOIN  <- sinon : +197 % (b.8)│
 │ alerte sur la DERIVE DE L'HYPOTHESE (normalization.invalidated_if, W42),   │
 │ pas sur l'erreur : l'erreur bouge aussi quand tout va bien.                │
 └────────────────────────────────────────────────────────────────────────────┘
```
 
### `forecast_service.yaml`
 
```yaml
version: 1
asof: 2026-10-25
service:
  name: forecast-park-hourly
  protocol: open-inference-protocol/v2
  # L'OIP V2 n'a AUCUN champ temporel (verifie 25/10/2026). Tout ce qui suit
  # vit dans la passerelle, PAS dans le protocole. Le declarer evite de croire
  # que le protocole le garantit.
  protocol_cannot_express: [as_of, horizon, quantile_levels, training_asof,
                            feature_version, cache_ttl]
  gateway_adds: [as_of, horizon, tau, training_asof, model_version,
                 feature_version]
data:
  closing_lag_steps: 48                  # d : latence de cloture, MESUREE
  backtest_must_use_same_lag: true       # sinon +6,3 % d'illusion (bloc 2)
  fail_if_window_not_closed: true
freshness:                               # budget INDEXE SUR LE QUANTILE SERVI
  budget_by_tau:
    "0.50": {max_model_age_steps: 168}   # mesure : k_max = 168
    "0.80": {max_model_age_steps: 120}   # mesure : k_max = 120
    "0.95": {max_model_age_steps: 24}    # mesure : k_max = 24  <- facteur 7
  on_budget_exceeded: refuse_and_fallback_to_seasonal_naive
stability:                               # la clause SIGNABLE (bloc 5)
  metric: relative_revision_vs_mase_scale
  jump_p95_max: 0.10
  visible_revision_threshold: 0.01
  max_pct_origins_visible: 0.15
  note: >
    jump_p95_max et max_pct_origins_visible tirent en sens OPPOSES. Mesure :
    du continu au mensuel, le saut p95 est multiplie par 223 et sa frequence
    divisee par 126. On en signe UN, pas les deux.
retraining:
  trigger: cusum_on_realized_error       # 40,9 % de ciblage contre 12,2 %
  forbid_trigger_on_future_error: true   # cf. PILOT : la pseudo-etiquette est
                                         # future, le declencheur deploye ne lit
                                         # que des erreurs CLOSES
  window_length_is_not_cadence: true     # w* = 27 pour k = 1, 24 et 168
  holdout:
    permanent: true
    fraction: 0.05                       # <- sans lui, l'effet est inmesurable
    rationale: "placebo naif a 197 % de l'effet reel (bloc 8)"
  forbid_drift_recall_as_kpi: true       # 100 % de rappel, 3e pire MASE
serving:
  cache_key: [series_id, as_of, horizon, tau, model_version, feature_version]
  ttl_ms_by_horizon: {"1": 300000, "24": 3600000}
  compute_window:
    nightly_minutes: 30
    required_throughput_inferences_s: auto   # S / fenetre
    provisioning_from_published_latency: true
model_registry:
  weights_license_must_be_read_at_source: true
  allowed_weight_licenses: [apache-2.0, mit, bsd-3-clause, cc-by-4.0]
  repo_license_is_not_weights_license: true   # Timestral ET Chronos-2
  forbidden_ids: ["google/timesfm-3.0"]       # W42
  pending_license_review: ["mistralai/Timestral-*", "amazon/chronos-2"]
versioning:
  pinned_together: [model_id, feature_version, training_asof, window_length,
                    cadence, guardrail_set_sha]
  results_incomparable_across: [closing_lag_steps, mase_scale_definition]
fail_mode: fail_closed
```
 
### Trois arbitrages explicites
 
| # | Arbitrage | Option A | Option B | **Décision** | Critère |
|---|---|---|---|---|---|
| 1 | Où vit la sémantique temporelle | dans `parameters` de l'OIP, par convention de serveur | dans une **passerelle de prévision** devant le runtime | **B** | La spécification dit « Currently no parameters are defined » : tout ce qu'on y met est non portable. Et sans `as_of` dans la **clé de cache**, deux requêtes émises à deux dates sont confondues. Le MCP a rendu `ttlMs`/`cacheScope` obligatoires en juillet 2026 ; l'OIP n'a pas d'équivalent. |
| 2 | Déclencheur de réentraînement | détecteur de dérive (le réflexe) | **CUSUM sur erreur réalisée + budget de fraîcheur par quantile** | **B** | Mesuré : le détecteur **parfait** fait 100 % de réentraînements utiles et **MASE 1,3685 contre 0,9829** pour un cron hebdomadaire. Le CUSUM, lui, cible **40,9 %** contre 12,2 % à budget comparable et gagne dès que la stabilité est tarifée. **La dérive détectable n'est pas la dérive coûteuse.** |
| 3 | Fraction témoin | 0 % (on réentraîne tout ce qui déclenche) | **5 % de témoin permanent** | **B** | Sans témoin, le DiD et l'A/B sont **incalculables par construction** et l'estimateur naïf rend **197 %** de l'effet réel — dont **66 % de régression vers la moyenne pure**. Coût mesuré du témoin à 5 % : **0,0057** d'erreur supplémentaire sur le parc, pour un IC exploitable (demi-largeur 0,0352 contre un effet de 0,1053). **Un chiffre qui veut dire quelque chose coûte 5 % de service.** |
 
**Le piège d'architecture à nommer en entretien :** `freshness.budget_by_tau`. Un SLA de fraîcheur unique (« le modèle
a moins d'une semaine ») est **faux pour la moitié du service** : mesuré au bloc 5, le quantile 0,95 se dégrade
**+97 %** entre la cadence continue et la mensuelle quand la médiane ne se dégrade que de **+41 %**, et la cadence
maximale où le modèle bat encore la naïve saisonnière passe de **168 pas à τ = 0,50** à **24 pas à τ = 0,95**.
**Le budget de fraîcheur est une fonction du quantile servi. Un SLA scalaire est une erreur de type.**
 
---
 
## 4 — Fondamentaux : stats, maths & causalité · 20 min
 
### La notion : la **stabilité** d'un service de prévision, et sa décomposition en bruit et information
 
C'est la notion que le SMQC d'arXiv 2506.05776 nomme sans la fermer analytiquement. On la ferme ici.
 
**Modèle.** `y_t = L_t + e_t`, niveau `L_t = L_{t−1} + u_t` avec `u ~ N(0, q)` (marche aléatoire), bruit
`e ~ N(0, σ²)`. On sert `μ̂` = moyenne glissante de fenêtre `w`, **réestimée tous les `k` pas**. L'âge du modèle
déployé est `a`, de moyenne `(k−1)/2`.
 
**1. Exactitude.** L'erreur de prévision à un pas se décompose en bruit de la cible, erreur d'estimation du niveau sur
la fenêtre, **retard de centroïde de la fenêtre**, et **âge du modèle** :
 
```
    MSE(k, w) = σ²(1 + 1/w) + q·[ (ā + 1) + (w−1)(2w−1)/(6w) ],    ā = (k−1)/2
```
 
Le terme `(w−1)(2w−1)/(6w)` vient du calcul exact `Var[(1/w)Σ_j Σ_{i} u_i] = (q/w²)Σ_{j,j'}(a+1+min(j,j'))` avec
`Σ_{j,j'} min(j,j') = (w−1)w(2w−1)/6`. **Une moyenne de fenêtre `w` n'estime pas le niveau en `t` : elle l'estime au
centroïde de la fenêtre, soit `(w−1)/2` pas dans le passé.**
 
**Optimum de fenêtre, en forme close :**
 
```
    d/dw [ σ²/w + q(w−1)(2w−1)/(6w) ] = 0   ⟹   w* = √( 3(σ² + q/6) / q )  ≈  σ√(3/q)
```
 
**Et `w*` ne dépend pas de `k`.** La longueur de fenêtre et la cadence règlent **deux termes additifs indépendants**.
C'est l'erreur de raisonnement la plus coûteuse du domaine : on resserre le cron pour corriger un biais de fenêtre.
 
**2. Stabilité.** La variance du **saut publié** entre deux réestimations espacées de `k` pas, avec
`a = min(k,w)`, `b = max(k,w)` :
 
```
    Var(R) = 2·a·σ²/w²                                  ← jitter d'ESTIMATION  (du bruit)
           + (q/w²)·[ a²(b−a+1) + (a−1)a(2a−1)/3 ]      ← DERIVE reelle       (de l'information)
```
 
Le second terme vient du comptage exact des multiplicités `c_m` de chaque incrément `u_m` dans la différence des deux
moyennes (rampe 1…a, plateau à `a`, rampe descendante), puis de `Var = (q/w²)Σ c_m²`.
 
**3. Et la décomposition est le livrable.** `part_bruit = Var_estim / Var(R)` est **la fraction de la variance de vos
révisions publiées qui n'est que du bruit d'estimation**. C'est une clause de service mesurable et signable, et
personne ne la calcule.
 
**Intuition.** Publier une révision, c'est affirmer « l'information a changé ». La partie dérive **est** de
l'information : le niveau a réellement bougé, il faut le dire. La partie jitter est **du bruit qu'on republie**. À
cadence continue sur une fenêtre de 240, **67,6 % de la variance publiée est du bruit** : deux tiers des révisions
servies à un client ne portent aucune information.
 
**Le cas où ça casse : deux régimes de churn, et dans le second la cadence n'est plus un levier.**
- régime **bruit** (`k` petit) : `E|R| ∝ √k·σ/w`, donc `churn/pas = E|R|/k ∝ 1/√k`. **Diviser la fréquence par 100 ne
  divise le churn que par 10** : la cadence est un levier faible.
- régime **dérive** (`k` grand, `k ≤ w`) : `Var(R) ≈ q·k²/w`, donc `E|R| ∝ k·√(q/w)` et `churn/pas → √(q/w)·√(2/π)`,
  **une constante**. **La cadence n'est plus un levier du tout** : le niveau a bougé, il faudra le publier.
### Code — **exécuté, sorties réelles**
 
```python
"""Bloc 4 - W44 : la STABILITE d'un service de prevision, en forme close.
 
Modele. y_t = L_t + e_t, L_t marche aleatoire d'increment u ~ N(0, q),
e ~ N(0, sigma^2). On sert mu_hat = moyenne glissante de fenetre w, reestimee
tous les k pas. Age du modele deploye : a, de moyenne (k-1)/2.
 
Trois quantites, dont la troisieme n'est jamais mesuree :
 
  1. MSE(k, w) = sigma^2 (1 + 1/w) + q[ (a+1) + (w-1)(2w-1)/(6w) ]
     Le dernier terme est le RETARD DE CENTROIDE de la fenetre. Optimum
     interieur : w* = sqrt(3(sigma^2 + q/6)/q) ~ sigma sqrt(3/q), INDEPENDANT
     de la cadence k.
 
  2. Var(revision) entre deux reestimations espacees de k pas, avec
     a = min(k,w), b = max(k,w) :
         Var(R) = 2 a sigma^2 / w^2                      <- jitter d'ESTIMATION
                + (q/w^2)[ a^2 (b-a+1) + (a-1)a(2a-1)/3 ] <- DERIVE reelle
     La decomposition est le livrable : la premiere part est du BRUIT publie,
     la seconde est de l'INFORMATION. Seule la premiere se negocie.
 
  3. churn par pas = E|R| / k. Deux regimes, et c'est la ou la pratique se
     trompe : en regime de bruit, churn ~ 1/sqrt(k) (levier faible) ; en
     regime de derive, churn -> sqrt(q/w), CONSTANT (plus aucun levier).
 
Dependances : numpy seul.
"""
from __future__ import annotations
 
import numpy as np
 
SQ2PI = np.sqrt(2.0 / np.pi)
 
 
def centroid_lag_var(w: int) -> float:
    """Coefficient du retard de centroide : (w-1)(2w-1)/(6w), a multiplier par q."""
    return (w - 1) * (2 * w - 1) / (6.0 * w)
 
 
def mse_closed(k: int, w: int, sigma: float, q: float) -> float:
    a_bar = (k - 1) / 2.0
    return sigma ** 2 * (1 + 1 / w) + q * ((a_bar + 1) + centroid_lag_var(w))
 
 
def w_star(sigma: float, q: float) -> float:
    return float(np.sqrt(3.0 * (sigma ** 2 + q / 6.0) / q))
 
 
def var_revision_parts(k: int, w: int, sigma: float, q: float) -> tuple[float, float]:
    """Rend (part de jitter d'estimation, part de derive reelle)."""
    a, b = min(k, w), max(k, w)
    noise = 2.0 * a * sigma ** 2 / w ** 2
    drift = q * (a ** 2 * (b - a + 1) + (a - 1) * a * (2 * a - 1) / 3.0) / w ** 2
    return noise, drift
 
 
def var_revision(k, w, sigma, q):
    return sum(var_revision_parts(k, w, sigma, q))
 
 
def churn_per_step(k, w, sigma, q):
    return SQ2PI * np.sqrt(var_revision(k, w, sigma, q)) / k
 
 
def noise_share(k, w, sigma, q):
    """Fraction de la VARIANCE des revisions publiees qui n'est que du bruit
    d'estimation. C'est la clause de service qu'on peut signer."""
    n, d = var_revision_parts(k, w, sigma, q)
    return n / (n + d)
 
 
def mc(k, w, sigma, q, T=300_000, seed=44):
    rng = np.random.default_rng(seed)
    L = np.cumsum(rng.normal(0, np.sqrt(q), T))
    y = L + rng.normal(0, sigma, T)
    se, revs, mu = [], [], None
    for t in range(w, T - 1):
        if (t - w) % k == 0:
            new = y[t - w + 1:t + 1].mean()
            if mu is not None:
                revs.append(new - mu)
            mu = new
        se.append((y[t + 1] - mu) ** 2)
    return float(np.mean(se)), float(np.var(revs)), float(np.mean(np.abs(revs)))
 
 
if __name__ == "__main__":
    SIG, Q = 1.0, 0.004
    print(f"sigma = {SIG}, q = {Q}  ->  w* = sqrt(3(sigma^2+q/6)/q) = "
          f"{w_star(SIG, Q):.1f}\n")
 
    print("1) validation Monte-Carlo des deux formes closes (300 000 pas)")
    print(f"{'k':>5s} {'w':>5s} {'MSE close':>10s} {'MSE MC':>9s} {'ecart':>7s} "
          f"{'Var(R) close':>13s} {'Var(R) MC':>11s} {'ecart':>7s} "
          f"{'E|R| close':>11s} {'E|R| MC':>9s}")
    for k, w in ((1, 20), (1, 100), (10, 100), (50, 100), (100, 100),
                 (200, 100), (24, 240), (168, 240)):
        m, vr, ar = mc(k, w, SIG, Q)
        mc_, vc = mse_closed(k, w, SIG, Q), var_revision(k, w, SIG, Q)
        print(f"{k:>5d} {w:>5d} {mc_:>10.4f} {m:>9.4f} {mc_/m-1:>+6.1%} "
              f"{vc:>13.6f} {vr:>11.6f} {vc/vr-1:>+6.1%} "
              f"{SQ2PI*np.sqrt(vc):>11.4f} {ar:>9.4f}")
 
    print("\n2) l'optimum de FENETRE est interieur et independant de la cadence")
    print(f"{'w':>6s} " + " ".join(f"{'MSE k='+str(k):>11s}" for k in (1, 24, 168)))
    for w in (10, 20, 27, 28, 40, 100, 240, 720):
        print(f"{w:>6d} " + " ".join(
            f"{mse_closed(k, w, SIG, Q):>11.4f}" for k in (1, 24, 168)))
    for k in (1, 24, 168):
        print(f"   argmin numerique (k={k:>3d}) = "
              f"{min(range(3, 600), key=lambda w: mse_closed(k, w, SIG, Q))}", end="")
    print(f"   |  forme close w* = {w_star(SIG, Q):.1f}")
 
    print("\n3) la DECOMPOSITION du churn : quelle part de vos revisions est du bruit ?")
    print(f"{'k':>6s} {'E|R| (saut)':>12s} {'churn/pas':>11s} {'x saut':>8s} "
          f"{'x churn':>9s} {'part BRUIT':>11s}")
    bj = SQ2PI * np.sqrt(var_revision(1, 240, SIG, Q))
    bc = churn_per_step(1, 240, SIG, Q)
    for k in (1, 2, 4, 24, 48, 168, 720):
        j = SQ2PI * np.sqrt(var_revision(k, 240, SIG, Q))
        print(f"{k:>6d} {j:>12.5f} {churn_per_step(k,240,SIG,Q):>11.6f} "
              f"{j/bj:>7.1f}x {churn_per_step(k,240,SIG,Q)/bc:>8.3f}x "
              f"{noise_share(k,240,SIG,Q):>10.1%}")
    print(f"   -> a k=1, {noise_share(1,240,SIG,Q):.0%} de la variance publiee est du "
          f"BRUIT d'estimation : on republie du bruit tous les pas.")
    print(f"   -> a k=168, elle tombe a {noise_share(168,240,SIG,Q):.1%} : le saut est "
          f"gros mais il porte de l'INFORMATION.")
    print(f"   -> plancher de churn (regime derive) = sqrt(q/w)*sqrt(2/pi) = "
          f"{SQ2PI*np.sqrt(Q/240):.6f} contre churn a k=720 = "
          f"{churn_per_step(720,240,SIG,Q):.6f}")
 
    print("\n4) objectif de service J = MSE + lam * churn/pas  (w = 240 fixe)")
    print(f"{'lam':>8s} {'k optimal':>10s} {'J':>9s} {'MSE':>8s} {'churn/pas':>11s} "
          f"{'part BRUIT':>11s}")
    for lam in (0.0, 1.0, 10.0, 100.0, 1000.0, 10_000.0):
        kb = min(range(1, 721), key=lambda k: mse_closed(k, 240, SIG, Q)
                 + lam * churn_per_step(k, 240, SIG, Q))
        print(f"{lam:>8.0f} {kb:>10d} "
              f"{mse_closed(kb,240,SIG,Q)+lam*churn_per_step(kb,240,SIG,Q):>9.4f} "
              f"{mse_closed(kb,240,SIG,Q):>8.4f} "
              f"{churn_per_step(kb,240,SIG,Q):>11.6f} "
              f"{noise_share(kb,240,SIG,Q):>10.1%}")
    print("   et le meme objectif avec w LIBRE (le vrai arbitrage) :")
    print(f"{'lam':>8s} {'k*':>6s} {'w*':>6s} {'J':>9s} {'MSE':>8s} {'part BRUIT':>11s}")
    for lam in (0.0, 10.0, 100.0, 1000.0):
        best = min(((k, w) for k in range(1, 400) for w in range(10, 500, 5)),
                   key=lambda kw: mse_closed(*kw, SIG, Q)
                   + lam * churn_per_step(*kw, SIG, Q))
        k, w = best
        print(f"{lam:>8.0f} {k:>6d} {w:>6d} "
              f"{mse_closed(k,w,SIG,Q)+lam*churn_per_step(k,w,SIG,Q):>9.4f} "
              f"{mse_closed(k,w,SIG,Q):>8.4f} {noise_share(k,w,SIG,Q):>10.1%}")
 
    print("\n5) LE CONTRE-EXEMPLE : sous TENDANCE, la cadence ne corrige pas le biais")
    b, w = 0.01, 240
    print(f"   y_t = b*t + bruit, b = {b}, w = {w}. Une moyenne de fenetre w estime")
    print(f"   le niveau au CENTROIDE, soit (w-1)/2 = {(w-1)/2:.1f} pas dans le passe.")
    print(f"   plancher incompressible = b*[(w-1)/2 + 1] = {b*((w-1)/2+1):.3f}")
    print(f"{'k':>6s} {'biais theorique':>16s} {'biais MC':>10s} "
          f"{'% du biais k=720':>18s} {'part = fenetre':>15s}")
    rng = np.random.default_rng(7)
    T = 60_000
    y = b * np.arange(T) + rng.normal(0, 1.0, T)
    ref = b * ((w - 1) / 2 + (720 - 1) / 2 + 1)
    for k in (1, 24, 168, 720):
        err, mu = [], None
        for t in range(w, T - 1):
            if (t - w) % k == 0:
                mu = y[t - w + 1:t + 1].mean()
            err.append(y[t + 1] - mu)
        th = b * ((w - 1) / 2 + (k - 1) / 2 + 1)
        print(f"{k:>6d} {th:>16.3f} {np.mean(err):>10.3f} {th/ref:>17.1%} "
              f"{b*((w-1)/2+1)/th:>14.1%}")
    print(f"\n   -> passer de mensuel a continu retire "
          f"{1 - b*((w-1)/2+1)/ref:.0%} du biais, mais il RESTE "
          f"{b*((w-1)/2+1):.3f}, dont 99 % est le biais de FENETRE.")
    print("      Aucune cadence ne le touche. Le correctif est un terme de")
    print("      tendance ou une ponderation, pas un cron plus frequent.")
```
 
**Sorties réelles (σ = 1, q = 0,004 ⟹ `w* = 27,4`) :**
 
**1) Validation Monte-Carlo (300 000 pas) — les deux formes closes sont exactes :**
 
| k | w | MSE close | MSE MC | écart | Var(R) close | Var(R) MC | écart |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 1,0787 | 1,0752 | +0,3 % | 0,005200 | 0,005178 | +0,4 % |
| 1 | 100 | 1,1453 | 1,1440 | +0,1 % | 0,000240 | 0,000240 | +0,2 % |
| 10 | 100 | 1,1633 | 1,1622 | +0,1 % | 0,005868 | 0,005924 | −0,9 % |
| 50 | 100 | 1,2433 | 1,2436 | −0,0 % | 0,093340 | 0,094893 | −1,6 % |
| 100 | 100 | 1,3433 | 1,3437 | −0,0 % | 0,286680 | 0,290087 | −1,2 % |
| 200 | 100 | 1,5433 | 1,5422 | +0,1 % | 0,686680 | 0,683307 | +0,5 % |
| 24 | 240 | 1,3722 | 1,3666 | +0,4 % | 0,010114 | 0,009971 | +1,4 % |
| 168 | 240 | 1,6602 | 1,6499 | +0,6 % | 0,366477 | 0,360026 | +1,8 % |
 
**2) `argmin_w MSE = 27` pour `k = 1`, `k = 24` ET `k = 168`** — identique aux trois cadences, contre une forme close
`w* = 27,4`. **L'optimum de fenêtre est indépendant de la cadence.**
 
**3) La décomposition du churn (w = 240) :**
 
| k | E\|R\| (saut) | churn/pas | × saut vs k=1 | × churn vs k=1 | **part BRUIT** |
|---|---|---|---|---|---|
| 1 | 0,00572 | 0,005720 | 1,0× | 1,000× | **67,6 %** |
| 2 | 0,00930 | 0,004652 | 1,6× | 0,813× | 51,1 % |
| 4 | 0,01604 | 0,004010 | 2,8× | 0,701× | 34,4 % |
| 24 | 0,08024 | 0,003343 | 14,0× | 0,585× | 8,2 % |
| 48 | 0,15453 | 0,003219 | 27,0× | 0,563× | 4,4 % |
| 168 | 0,48302 | 0,002875 | 84,4× | 0,503× | 1,6 % |
| 720 | 1,27869 | 0,001776 | **223,6×** | **0,310×** | **0,3 %** |
 
→ **`k × 720` multiplie le saut par 223,6 et ne divise le churn par pas que par 3,2.** Et le plancher du régime dérive
vaut `√(q/w)·√(2/π) = 0,003257` : à `k ≥ 48` on est **déjà au plancher**, toute cadence plus lente ne fait plus que
grossir les sauts. *(À `k = 720 > w = 240` les fenêtres sont disjointes et le churn/pas tombe sous le plancher à
0,001776 : on ne suit plus la dérive du tout, on reste simplement faux.)*
 
**4) Objectif de service `J = MSE + λ·churn/pas` — et l'enseignement est dans la seconde table :**
 
| λ | k* (w = 240 figé) | part bruit | | λ | k* (w libre) | **w*** | part bruit |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 67,6 % | | 0 | 1 | **30** | 94,3 % |
| 10 | 4 | 34,4 % | | 10 | 11 | **60** | 44,6 % |
| 100 | 13 | 14,0 % | | 100 | 18 | **175** | 14,1 % |
| 1 000 | 429 | 0,6 % | | 1 000 | 30 | **495** | 3,3 % |
 
→ **Quand on tarife la stabilité et qu'on laisse la fenêtre libre, l'optimum allonge la FENÊTRE bien plus qu'il ne
ralentit la cadence** : à λ = 1 000, `k*` passe de 429 (fenêtre figée) à **30**, pour une fenêtre de **495**. Et `J`
tombe de 4,3862 à **4,0054**. **Le bon levier de stabilité est la longueur de fenêtre ; le cron est le mauvais
levier, et c'est celui sur lequel tout le monde appuie.**
 
**5) LE CONTRE-EXEMPLE : sous tendance, la cadence ne corrige pas le biais.**
`y_t = b·t + bruit`, `b = 0,01`, `w = 240`. Plancher incompressible `b·[(w−1)/2 + 1] = 1,205`.
 
| k | biais théorique | biais MC | % du biais de k=720 | part due à la FENÊTRE |
|---|---|---|---|---|
| 1 | 1,205 | 1,205 | 25,1 % | **100,0 %** |
| 24 | 1,320 | 1,320 | 27,5 % | 91,3 % |
| 168 | 2,040 | 2,039 | 42,5 % | 59,1 % |
| 720 | 4,800 | 4,797 | 100,0 % | 25,1 % |
 
→ Passer du mensuel au continu retire **75 %** du biais. **Mais 100 % de ce qui reste est le biais de centroïde de
fenêtre, et aucune cadence ne le touche** — à la limite `k → 1` il reste `1,205`. Le correctif est **un terme de
tendance ou une pondération**, pas un cron plus fréquent. **C'est la réponse à la demande la plus fréquente qu'un
client formule : « on sous-estime depuis six mois, on va réentraîner plus souvent. »**
 
### Question d'entretien senior
> *« Votre prévision est systématiquement sous le réalisé depuis six mois sur un parc en croissance. L'équipe propose
> de passer d'un réentraînement mensuel à quotidien. Qu'est-ce que vous répondez ? »*
 
**Réponse attendue, en trois temps chiffrés.**
1. **Je décompose le biais avant de toucher au cron.** Une moyenne de fenêtre `w` estime le niveau au centroïde, soit
   `(w−1)/2` pas dans le passé. Le biais total vaut `b·[(w−1)/2 + (k−1)/2 + 1]`. Avec `w = 240`, passer de `k = 720`
   à `k = 1` retire **75 %** du biais — mais il **reste `b·(w−1)/2`**, et c'est **100 % du résidu**. Donc oui, ça
   aidera ; non, ça ne réglera pas le problème ; et je peux dire d'avance de combien.
2. **Le vrai correctif est un terme de tendance, ou une fenêtre plus courte.** Et la fenêtre a un optimum intérieur,
   `w* = √(3(σ² + q/6)/q)`, **indépendant de `k`** : je peux le calculer avant d'avoir réentraîné une seule fois.
3. **Et je chiffre ce que la cadence quotidienne coûte en stabilité.** `k` divisé par 30 multiplie la fréquence des
   révisions publiées par 30 et fait monter la **part de bruit** dans la variance publiée de 8,2 % (`k = 24`) à
   **67,6 %** (`k = 1`) sur une fenêtre de 240. Donc : **on va republier du bruit aux deux tiers pour corriger un
   quart d'un biais dont les trois quarts restants ne viennent pas de la cadence.** Je propose l'ordre inverse : terme
   de tendance d'abord, fenêtre ensuite, cron en dernier.
**Le piège de la question :** répondre « oui, réentraînons plus souvent » (ça marche un peu, donc le symptôme
disparaît partiellement et le diagnostic est enterré), ou « non, c'est inutile » (c'est faux : 75 %). **Le senior
décompose, chiffre les deux termes séparément, et classe les correctifs par rapport coût/effet.**
 
---
 
## 5 — Mini-projet de la semaine · 40 min
 
### Objectif métier (1 phrase)
**Dire à un opérateur, en euros et par écrit, à quelle cadence réentraîner son parc de prévisions et quelle clause de
stabilité il peut signer — et lui prouver que le SLA de fraîcheur unique qu'il a déjà écrit est inatteignable pour la
moitié de son service.**
 
### Jeu de données
 
| | |
|---|---|
| Jeu de référence | **Electricity Hourly Dataset** (Monash Time Series Forecasting Archive) |
| Lien | **[zenodo.org/records/4656140](https://zenodo.org/records/4656140)** *(page vérifiée le 25/10/2026)* |
| Volumétrie | **321 séries horaires**, 2012 → 2014, archive ZIP de **11,8 Mo**, format `.tsf` |
| Granularité / horizon | horaire ; ~26 000 pas par série, soit **3 ans** — assez long pour un backtest de cadence mensuelle |
| Licence | **CC BY 4.0** — usage portfolio et commercial autorisé avec attribution |
| DOI | 10.5281/zenodo.4656140 |
| ⚠️ Piège de la fiche | `zenodo.org/records/4656141` est l'Electricity **WEEKLY** (fiche W41, reconduite) |
 
> ⚠️ **Honnêteté d'exécution, et c'est une information d'environnement neuve.** Le **téléchargement direct du fichier**
> depuis Zenodo est **bloqué par le proxy de cette session (`CONNECT tunnel failed, 403`)**, alors que la **page** du
> dépôt se lit parfaitement par WebFetch. Les chiffres ci-dessous sont donc produits sur un **DGP calibré** de même
> granularité (horaire), même saisonnalité (`M = 24`), avec ruptures de niveau **et** dérive d'amplitude saisonnière.
> **Pour la version client, le parseur `.tsf` durci de W41 (`tsf_parse.py`, 6 assertions, déjà réutilisé tel quel en
> W42) se branche directement** : c'est une substitution de source, pas une réécriture.
 
### Plan en 6 étapes chronométrées
 
| # | Étape | min | Contenu |
|---|---|---|---|
| 1 | Chaîne fraîcheur → quantile → euros | **8** | Le modèle sert `μ̂ + profil + z_τ·σ̂` ; perte = **pinball à τ**. Trois niveaux de service : `τ ∈ {0,50 ; 0,80 ; 0,95}`. **Piège à ne pas rater : `σ̂` se calcule sur le résidu APRÈS retrait du profil saisonnier.** |
| 2 | Baseline obligatoire | **5** | **Naïve saisonnière probabiliste** (`y_{t−24}` + `z_τ ×` écart-type des différences saisonnières sur la fenêtre). Sans réentraînement possible : c'est le plancher qu'il faut battre. |
| 3 | Balayage de cadence | **8** | `k ∈ {1, 12, 24, 36, 48, 72, 120, 168, 336, 720}`, rolling origin, latence de clôture `d = 48` appliquée partout. |
| 4 | La décision en euros | **8** | `perte = pinball × volume × marge` + `réentraînements × prix de calcul`. Balayage du **prix du calcul** : 0,05 / 0,50 / 2,00 / 10,00 € par série réentraînée. |
| 5 | **Le vrai livrable : la triple contrainte** | **7** | Trois cadences maximales, calculées séparément : exactitude par quantile, clause de stabilité, optimum économique. **Le rapport dit laquelle on optimise.** |
| 6 | Artefact + REX | **4** | `forecast_service.yaml` rempli, 2 graphiques, et la ligne qui se facture : l'écart entre les trois cadences. |
 
### Baselines obligatoires à battre
 
| Famille | Baseline | Métrique | Protocole |
|---|---|---|---|
| Plancher | **naïve saisonnière probabiliste** | pinball à τ = 0,50 / 0,80 / 0,95 | rolling origin, `d = 48`, test figé |
| Référence de cadence | **calendaire mensuel (`k = 720`)** — ce que fait le client | idem + nombre de réentraînements | idem |
| Honnêteté | **le modèle gelé (`jamais`)** | idem | doit être dans le tableau. MASE 1,6003 au bloc 2 : si une politique ne le bat pas, elle n'existe pas. |
| Interdit | **classer sur la MASE seule** | — | `forbid_ranking_on: [mase, mae, mape]` (W41) : ici c'est la **pinball au quantile servi** qui arbitre, et elle ne donne pas le même classement. |
 
### Code — squelette exécutable, **exécuté**
 
```python
"""Bloc 5 - W44 : "combien coute un jour de retard ?"
 
Transformer une cadence de reentrainement en EUROS, et en deduire la clause de
service signable. Le point qui surprend : la cadence optimale depend du NIVEAU
DE SERVICE, pas seulement des donnees. A tau = 0,50 et tau = 0,95 on ne
reentraine pas a la meme frequence, parce que le retard ne biaise pas les
quantiles de la meme facon.
 
Chaine : fraicheur -> biais de quantile -> perte pinball -> euros -> cadence.
Baseline obligatoire : naive saisonniere + reentrainement mensuel.
 
Dependances : numpy seul.
"""
from __future__ import annotations
 
import numpy as np
 
M, H, W, T, N = 24, 24, 240, 2400, 200
D_LAG = 48
WARM = W + M + D_LAG
TAUS = (0.50, 0.80, 0.95)
 
 
def make_park(n=N, T=T, seed=544, rate=1 / 600.0, jump=0.8, sigma=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    sb = rng.normal(0, 1.2, (n, M)); sb -= sb.mean(axis=1, keepdims=True)
    cps = rng.random((n, T)) < rate; cps[:, :WARM] = False
    level = 10.0 + np.cumsum(rng.normal(0, jump, (n, T)) * cps, axis=1)
    amp = 1.0 + 0.35 * np.sin(2 * np.pi * t / T + rng.uniform(0, 6.28, (n, 1)))
    y = level + sb[:, t % M] * amp + rng.normal(0, sigma, (n, T))
    return dict(y=y, sigma=sigma)
 
 
def fit(y, t_end, w=W):
    yw = y[:, t_end - w + 1:t_end + 1]
    mu = yw.mean(axis=1)
    r = yw - mu[:, None]
    ph = np.arange(t_end - w + 1, t_end + 1) % M
    prof = np.empty((y.shape[0], M))
    for k in range(M):
        prof[:, k] = r[:, ph == k].mean(axis=1)
    # dispersion APRES retrait du profil saisonnier. Bug classique : garder le
    # profil dans le residu gonfle sigma et detruit les quantiles hauts.
    sd = (r - prof[:, ph]).std(axis=1)
    return mu, prof, sd
 
 
def pinball(y, q, tau):
    d = y - q
    return np.where(d >= 0, tau * d, (tau - 1) * d)
 
 
def run(park, k, taus=TAUS, fit_lag=D_LAG):
    """Rend, par tau : perte pinball moyenne, et la clause de stabilite."""
    y = park["y"]; n = y.shape[0]
    from scipy.stats import norm
    z = {t: float(norm.ppf(t)) for t in taus}
    mu, prof, sd = fit(y, WARM - fit_lag)
    loss = {t: np.zeros(n) for t in taus}
    revs = {t: [] for t in taus}
    prev = {t: None for t in taus}
    nre = 0; cnt = 0
    for t0 in range(WARM, T - H):
        ph = np.arange(t0 + 1, t0 + H + 1) % M
        base = mu[:, None] + prof[:, ph]
        act = y[:, t0 + 1:t0 + H + 1]
        for tau in taus:
            q = base + z[tau] * sd[:, None]
            loss[tau] += pinball(act, q, tau).mean(axis=1)
            if prev[tau] is not None:
                revs[tau].append(np.abs(q[:, :-1] - prev[tau][:, 1:]).mean(axis=1))
            prev[tau] = q
        cnt += 1
        if t0 > WARM and (t0 - WARM) % k == 0:
            mu, prof, sd = fit(y, t0 - fit_lag); nre += 1
    scale = np.abs(y[:, M:] - y[:, :-M]).mean(axis=1)
    out = {}
    for tau in taus:
        r = np.asarray(revs[tau]) / scale[None, :]
        out[tau] = dict(pinball=float((loss[tau] / cnt).mean()),
                        jump_p95=float(np.percentile(r[r > 1e-12], 95))
                        if (r > 1e-12).any() else 0.0,
                        pct_vis=float((r > 0.01).mean()))
    return out, nre
 
 
def snaive(park, taus=TAUS):
    """Baseline obligatoire : naive saisonniere, aucun reentrainement possible."""
    y = park["y"]; n = y.shape[0]
    loss = {t: np.zeros(n) for t in taus}; cnt = 0
    for t0 in range(WARM, T - H):
        idx = np.arange(t0 + 1, t0 + H + 1) - M
        q0 = y[:, idx]
        act = y[:, t0 + 1:t0 + H + 1]
        # dispersion empirique des differences saisonnieres, fenetre W
        d = y[:, t0 - W + 1:t0 + 1] - y[:, t0 - W + 1 - M:t0 + 1 - M]
        s = d.std(axis=1)
        from scipy.stats import norm
        for tau in taus:
            loss[tau] += pinball(act, q0 + norm.ppf(tau) * s[:, None], tau).mean(axis=1)
        cnt += 1
    return {t: float((loss[t] / cnt).mean()) for t in taus}
 
 
if __name__ == "__main__":
    park = make_park()
    print(f"parc : {N} series x {T} pas | H = {H} | W = {W} | latence d = {D_LAG}")
    print("Jeu de donnees de reference du livrable (lien verifie le 25/10/2026) :")
    print("  Monash Electricity Hourly, 321 series horaires, 2012-2014, CC-BY-4.0")
    print("  https://zenodo.org/records/4656140")
    print("  /!\\ telechargement direct BLOQUE par le proxy de cette session (403)")
    print("      => chiffres ci-dessous produits sur un DGP calibre, meme")
    print("         granularite (horaire), meme saisonnalite (M=24).\n")
 
    KS = (1, 12, 24, 36, 48, 72, 120, 168, 336, 720)
    base = snaive(park)
    print("baseline naive saisonniere (sans reentrainement) :",
          " ".join(f"tau={t:.2f} pinball {v:.4f}" for t, v in base.items()), "\n")
 
    print("1) perte pinball par cadence et par niveau de service")
    print(f"{'k (pas)':>9s} " + " ".join(f"{'tau='+f'{t:.2f}':>12s}" for t in TAUS)
          + f" {'reentr.':>9s}")
    res = {}
    for k in KS:
        o, nre = run(park, k); res[k] = (o, nre)
        print(f"{k:>9d} " + " ".join(f"{o[t]['pinball']:>12.4f}" for t in TAUS)
              + f" {nre:>9d}")
    print("   gain relatif par rapport a la naive saisonniere :")
    for k in KS:
        print(f"{k:>9d} " + " ".join(
            f"{1-res[k][0][t]['pinball']/base[t]:>11.1%}" for t in TAUS))
 
    print("\n2) LA DECISION EN EUROS")
    print("   perte = pinball x volume x marge_unitaire ; cout = reentr. x prix_calcul")
    VOL, MARGE = 1_000.0, 12.0     # unites/pas/serie, euros de marge unitaire
    HORIZON_MOIS = (T - WARM - H) / 720.0
    for PRIX in (0.05, 0.50, 2.00, 10.00):
        print(f"\n   prix d'un reentrainement de serie = {PRIX:.2f} EUR  "
              f"(parc {N} series, {HORIZON_MOIS:.1f} mois simules)")
        print(f"{'k':>8s} " + " ".join(f"{'tau='+f'{t:.2f}':>14s}" for t in TAUS))
        tot = {t: {} for t in TAUS}
        for k in KS:
            o, nre = res[k]
            for t in TAUS:
                eur = o[t]["pinball"] * VOL * MARGE / 100.0 * N * (T - WARM - H) / 1000
                eur_calc = nre * PRIX * N
                tot[t][k] = eur + eur_calc
            print(f"{k:>8d} " + " ".join(f"{tot[t][k]:>14,.0f}" for t in TAUS))
        print(f"{'optimum':>8s} " + " ".join(
            f"{'k='+str(min(tot[t], key=tot[t].get)):>14s}" for t in TAUS))
 
    print("\n3) LA CLAUSE DE STABILITE SIGNABLE (au niveau de service tau=0.95)")
    print(f"{'k':>8s} {'saut p95 (% echelle)':>22s} {'% origines >1%':>16s} "
          f"{'clause possible':>34s}")
    for k in KS:
        o, _ = res[k]
        s = o[0.95]
        cl = (f"revision <= {s['jump_p95']*100:.0f} % de l'echelle"
              if s["jump_p95"] < 0.5 else "AUCUNE clause d'amplitude tenable")
        print(f"{k:>8d} {s['jump_p95']*100:>21.1f} {s['pct_vis']:>15.1%} {cl:>34s}")
 
    print("\n4) LA TRIPLE CONTRAINTE : la cadence admissible n'est pas unique")
    print("   (a) cadence maximale ou le modele batt ENCORE la naive saisonniere,")
    print("       par niveau de service :")
    for t in TAUS:
        adm = [k for k in KS if res[k][0][t]["pinball"] < base[t]]
        print(f"       tau = {t:.2f} -> k_max = {max(adm) if adm else 0:>4d} "
              f"(degradation de k=1 a k=720 : "
              f"{res[720][0][t]['pinball']/res[1][0][t]['pinball']-1:+.0%})")
    print("   (b) cadence maximale compatible avec une clause 'revision p95 <= 10 %'")
    adm = [k for k in KS if res[k][0][0.95]["jump_p95"] <= 0.10]
    print(f"       -> k_max = {max(adm) if adm else 0}")
    print("   (c) cadence minimisant le cout total : voir le bloc 2 ci-dessus,")
    print("       elle depend du PRIX DU CALCUL, pas des donnees.")
 
    print("\n5) l'illusion de backtest due a la latence de donnees (tau=0.95)")
    for k in (24, 168):
        a, _ = run(park, k, fit_lag=D_LAG)
        b, _ = run(park, k, fit_lag=0)
        print(f"   k={k:>4d} : pinball reelle {a[0.95]['pinball']:.4f} contre "
              f"fuyante {b[0.95]['pinball']:.4f} -> illusion "
              f"{(a[0.95]['pinball']-b[0.95]['pinball'])/b[0.95]['pinball']:+.1%}")
```
 
### Résultats obtenus
 
Parc de **200 séries × 2 400 pas**, `H = 24`, `W = 240`, `d = 48`.
Baseline naïve saisonnière : pinball **0,2880** (τ = 0,50) · **0,2032** (τ = 0,80) · **0,0758** (τ = 0,95).
 
**1) Perte pinball par cadence, et gain relatif contre la naïve saisonnière :**
 
| k (pas) | τ=0,50 | τ=0,80 | τ=0,95 | réentr. | gain τ=0,50 | gain τ=0,80 | **gain τ=0,95** |
|---|---|---|---|---|---|---|---|
| 1 | 0,2544 | 0,1827 | 0,0736 | 2 063 | +11,7 % | +10,1 % | **+3,0 %** |
| 12 | 0,2561 | 0,1841 | 0,0746 | 171 | +11,1 % | +9,4 % | +1,7 % |
| 24 | 0,2579 | 0,1857 | 0,0757 | 85 | +10,4 % | +8,6 % | **+0,2 %** |
| 36 | 0,2598 | 0,1873 | 0,0767 | 57 | +9,8 % | +7,8 % | **−1,2 %** |
| 48 | 0,2616 | 0,1889 | 0,0778 | 42 | +9,2 % | +7,0 % | −2,6 % |
| 72 | 0,2652 | 0,1922 | 0,0802 | 28 | +7,9 % | +5,4 % | −5,7 % |
| 120 | 0,2726 | 0,1990 | 0,0849 | 17 | +5,3 % | +2,0 % | −11,9 % |
| 168 | 0,2795 | 0,2046 | 0,0888 | 12 | **+3,0 %** | −0,7 % | −17,1 % |
| 336 | 0,3039 | 0,2259 | 0,1047 | 6 | −5,5 % | −11,2 % | −38,1 % |
| 720 | 0,3594 | 0,2782 | 0,1453 | 2 | −24,8 % | −36,9 % | **−91,6 %** |
 
**2) La décision en euros** (volume 1 000 unités/pas/série, marge 12 €/unité, 2,9 mois simulés, parc 200 séries) :
 
| prix d'un réentraînement | cadence optimale | coût total à l'optimum (τ=0,95) |
|---|---|---|
| **0,05 €** | **k = 48** | 4 275 € |
| **0,50 €** | **k = 168** | 5 601 € |
| **2,00 €** | **k = 336** | 7 585 € |
| **10,00 €** | **k = 720** | 11 197 € |
 
→ **L'optimum économique est piloté par le prix du calcul, et par rien d'autre** : il est **identique aux trois
niveaux de service** à prix donné. Et il se déplace de **k = 48 à k = 720, un facteur 15, pour un prix de calcul
multiplié par 200.**
 
**3) La clause de stabilité signable (au niveau de service τ = 0,95) :**
 
| k | saut p95 (% de l'échelle) | % d'origines > 1 % | clause possible |
|---|---|---|---|
| 1 | **2,7 %** | 36,8 % | « révision ≤ 3 % de l'échelle » |
| 12 | 15,1 % | 8,3 % | « révision ≤ 15 % » |
| 24 | 24,3 % | 4,1 % | « révision ≤ 24 % » |
| 48 | 47,4 % | 2,0 % | « révision ≤ 47 % » |
| 72 | 67,7 % | 1,4 % | **aucune clause d'amplitude tenable** |
| 168 | 143,4 % | 0,6 % | aucune |
| 720 | **305,5 %** | 0,1 % | aucune |
 
**4) LE LIVRABLE : la triple contrainte, et elle est incompatible.**
 
| contrainte | cadence maximale admissible | ce qui la fixe |
|---|---|---|
| (a) battre encore la naïve saisonnière, **τ = 0,50** | **k ≤ 168** | dégradation de +41 % entre `k = 1` et `k = 720` |
| (a) battre encore la naïve saisonnière, **τ = 0,80** | **k ≤ 120** | dégradation de +52 % |
| (a) battre encore la naïve saisonnière, **τ = 0,95** | **k ≤ 24** | dégradation de **+97 %** |
| (b) clause « révision p95 ≤ 10 % de l'échelle » | **k = 1** | le saut p95 passe 10 % dès `k = 12` |
| (c) optimum économique à 2 €/réentraînement | **k = 336** | le prix du calcul |
 
> **Trois réponses séparées par deux ordres de grandeur, sur le même parc, le même modèle, la même semaine : `k = 1`,
> `k = 24`, `k = 336`.** Et ce sont trois contraintes **également légitimes**. La question « à quelle cadence
> réentraîner ? » n'a donc pas de réponse technique — elle a une réponse **contractuelle**, et celui qui ne l'écrit
> pas en choisit une par défaut : celle de son ordonnanceur.
 
**5) L'illusion de backtest due à la latence de clôture, au quantile servi :**
 
| k | pinball réelle (`d = 48`) | pinball fuyante (`d = 0`) | illusion |
|---|---|---|---|
| 24 | 0,0757 | 0,0672 | **+12,5 %** |
| 168 | 0,0888 | 0,0811 | +9,6 % |
 
→ **L'illusion est deux fois plus grande au quantile 0,95 (+12,5 %) qu'en MASE (+6,3 %, bloc 2).** La latence de
clôture ne biaise pas la médiane et la queue de la même façon : elle frappe le quantile sur lequel repose la décision
de réapprovisionnement. **Même famille que W41 (« le quantile est la décision ») et W38 (point-in-time), au point de
jonction des deux.**
 
### Les quatre résultats à retenir
 
1. **Le budget de fraîcheur est une fonction du quantile servi, pas un scalaire.** `k_max` passe de **168 à τ = 0,50**
   à **24 à τ = 0,95** : **un facteur 7**, pour le même modèle et le même parc. Un SLA « le modèle a moins d'une
   semaine » est **vrai pour la médiane et faux de 7× pour le niveau de service**.
2. **Le quantile haut se dégrade 2,4× plus vite que la médiane sous l'effet du retard** (+97 % contre +41 % entre
   `k = 1` et `k = 720`). Mécanisme : le retard biaise le niveau, et un biais de niveau déplace la queue
   proportionnellement plus que le centre une fois normalisé par la perte pinball, dont le poids est `τ` du côté
   sous-estimé. **À τ = 0,95, sous-estimer coûte 19× plus que surestimer : le retard, qui est un biais de même signe
   que la croissance, tape exactement là.**
3. **À `k = 720`, le modèle est 91,6 % PIRE que la naïve saisonnière au quantile servi** alors qu'il n'est « que »
   24,8 % pire en médiane. **Un modèle réentraîné mensuellement sur un parc horaire est, au niveau de service, battu
   par `y_{t−24}`.** C'est le chiffre à montrer en premier rendez-vous.
4. **L'optimum économique et la clause de stabilité sont antagonistes.** À 0,05 €/réentraînement l'optimum est
   `k = 48`, dont le **saut p95 vaut 47,4 % de l'échelle** : économiquement optimal, **contractuellement insignable**.
### Definition of Done
- Les trois cadences maximales calculées **séparément** et nommées : exactitude par quantile, stabilité, économie.
- La courbe pinball × cadence pour les trois `τ`, avec la **naïve saisonnière tracée dessus** et les points de
  croisement marqués.
- Le tableau des coûts en euros pour au moins trois prix de calcul, et le prix de bascule entre deux cadences.
- Le tableau de stabilité `saut p95 / % d'origines visibles` et **la clause effectivement signable**.
- La mesure de l'illusion de latence de clôture **aux deux métriques** (MASE et pinball à τ servi).
- `forecast_service.yaml` rempli, avec `freshness.budget_by_tau` chiffré depuis les mesures.
- **Et la phrase du rapport :** « vous optimisez (c) ; votre contrat client promet (b) ; votre métrique de suivi
  mesure (a) à τ = 0,50. »
---
 
## 6 — IA générative & Agentic : expérimentation transverse · 20 min
 
### L'expérimentation
**Choisir la taille d'un foundation model de séries temporelles par le coût de serving, et non par le leaderboard** —
en n'utilisant **que les chiffres publiés** par Mistral le 15 octobre 2026 pour Timestral, et en calculant **le gain
d'exactitude qu'il faudrait** pour justifier chaque taille. Aucun score numérique n'est publié en texte sur la page
(ils ne sont que dans des images de graphiques) : **on ne les invente donc pas**, on calcule un seuil et on le compare
au seul chiffre publié, les **0,2 %** d'écart entre le 123M et le 209M.
 
C'est la suite directe du `breakeven.py` de W42 (rentabilité zero-shot vs local), prise par l'autre bout : **W42
demandait si un TSFM valait la peine d'être entraîné ; W44 demande quelle taille vaut la peine d'être servie.**
 
### Code — **exécuté**
 
```python
"""Bloc 6 - W44 : choisir la TAILLE d'un foundation model par le cout de
serving, et non par le leaderboard.
 
Toutes les latences viennent des chiffres PUBLIES par Mistral le 15/10/2026
pour Timestral (CPU, 8 threads, prevision a un jour depuis 3 semaines
d'historique au pas de 15 min) :
    5M -> 10 ms   |   209M -> 140 ms
    28M  = 5x plus rapide que 209M            -> 28 ms
    123M = 3/4 de la latence de 209M, et son score a 0,2 % pres -> 105 ms
AUCUN score numerique n'est publie en texte sur la page (ils ne sont que dans
des images de graphiques). On ne les invente donc PAS : on calcule le gain
d'exactitude qu'il FAUDRAIT pour justifier la taille, et on le compare au seul
chiffre publie, les 0,2 %.
 
Dependances : numpy seul.
"""
from __future__ import annotations
 
import numpy as np
 
# -- faits publies (verifies le 25/10/2026) --------------------------------- #
MODELS = {       # nom : (params M, latence ms CPU 8 threads, poids fp32 Mo)
    "Timestral-5M":   (5,    10.0,  21),
    "Timestral-28M":  (28,   28.0,  111),
    "Timestral-123M": (123, 105.0,  490),
    "Timestral-209M": (209, 140.0,  837),
}
PUBLISHED_GAIN_123_TO_209 = 0.002        # "within 0.2% of the 209M score"
WEIGHTS_LICENSE = "Modified MIT v1.0"    # NON Apache-2.0 ; le DEPOT est Apache-2.0
CODE_LICENSE = "Apache-2.0"
 
# -- parametres de serving (declares, pas publies) -------------------------- #
THREADS = 8
PRIX_VCPU_H = 0.035       # EUR par vCPU-heure
H_MOIS = 720              # heures par mois
 
 
def cout_compute(S: int, k: int, lat_ms: float,
                 prix=PRIX_VCPU_H, threads=THREADS) -> float:
    """EUR de calcul par mois pour S series rafraichies toutes les k heures."""
    n_inf = S * (H_MOIS / k)
    vcpu_s = n_inf * (lat_ms / 1000.0) * threads
    return vcpu_s / 3600.0 * prix
 
 
def breakeven_gain(S, k, lat_hi, lat_lo, c_loss_series_mois) -> float:
    """Gain RELATIF d'exactitude que le modele lent doit apporter pour que son
    surcout de calcul soit rembourse.
        delta* = (cout_hi - cout_lo) / (S * perte_decision_mensuelle_par_serie)
    """
    d = cout_compute(S, k, lat_hi) - cout_compute(S, k, lat_lo)
    return d / (S * c_loss_series_mois)
 
 
if __name__ == "__main__":
    print("=== A. ce que coute chaque taille, en euros de CPU par mois ===")
    print(f"prix {PRIX_VCPU_H} EUR/vCPU-h, {THREADS} threads, {H_MOIS} h/mois\n")
    for S in (1_000, 20_000, 500_000):
        print(f"  parc de {S:>7,} series")
        print(f"{'modele':>16s} " + " ".join(
            f"{'k='+str(k)+'h':>12s}" for k in (1, 6, 24, 168)))
        for nm, (p, lat, mo) in MODELS.items():
            print(f"{nm:>16s} " + " ".join(
                f"{cout_compute(S, k, lat):>12,.0f}" for k in (1, 6, 24, 168)))
        print()
 
    print("=== B. LA question de serving : 209M vaut-il mieux que 123M ? ===")
    print("surcout de latence = (140 - 105) / 105 = "
          f"{(140-105)/105:+.1%} de temps CPU, pour un gain PUBLIE de "
          f"{PUBLISHED_GAIN_123_TO_209:.1%}\n")
    print("gain d'exactitude NECESSAIRE pour rembourser le surcout "
          "(delta*), selon l'enjeu")
    print(f"{'perte decision':>16s} {'k=1h':>10s} {'k=6h':>10s} {'k=24h':>10s} "
          f"{'verdict a 0,2 %':>18s}")
    print(f"{'EUR/serie/mois':>16s}")
    S = 20_000
    for c in (0.10, 1.0, 10.0, 100.0, 1000.0):
        ds = [breakeven_gain(S, k, 140.0, 105.0, c) for k in (1, 6, 24)]
        verdict = ("209M justifie" if ds[1] <= PUBLISHED_GAIN_123_TO_209
                   else "209M DOMINE")
        print(f"{c:>16,.2f} " + " ".join(f"{d:>9.3%}" for d in ds)
              + f" {verdict:>18s}")
    c_crit = [c for c in np.logspace(-2, 4, 2000)
              if breakeven_gain(S, 6, 140.0, 105.0, c) <= PUBLISHED_GAIN_123_TO_209]
    print(f"\n  -> seuil (k=6h, parc {S:,}) : le 209M ne se justifie qu'au-dela de "
          f"{min(c_crit):.2f} EUR de perte de decision par serie et par mois.")
    print("     En dessous, les 0,2 % publies ne remboursent pas les +33 % de CPU.")
 
    print("\n=== C. et contre un modele CLASSIQUE, dont le serving est quasi gratuit ? ===")
    LAT_CLASSIQUE = 0.08      # ms : une moyenne glissante + profil, en numpy
    print(f"latence declaree du modele du bloc 2 (moyenne + profil) : "
          f"{LAT_CLASSIQUE} ms, soit {140/LAT_CLASSIQUE:,.0f}x moins que le 209M")
    print(f"{'enjeu EUR/serie/mois':>22s} {'delta* vs 209M':>16s} "
          f"{'delta* vs 28M':>15s}")
    for c in (0.10, 1.0, 10.0, 100.0):
        print(f"{c:>22,.2f} "
              f"{breakeven_gain(S, 6, 140.0, LAT_CLASSIQUE, c):>15.2%} "
              f"{breakeven_gain(S, 6, 28.0, LAT_CLASSIQUE, c):>14.2%}")
    print("  -> a 1 EUR/serie/mois, un TSFM doit battre la moyenne glissante de")
    print(f"     {breakeven_gain(S, 6, 140.0, LAT_CLASSIQUE, 1.0):.1%} (209M) ou "
          f"{breakeven_gain(S, 6, 28.0, LAT_CLASSIQUE, 1.0):.1%} (28M) pour etre rentable.")
    print("     C'est le 28M, pas le 209M, qui est le candidat serieux.")
 
    print("\n=== C2. LA CONTRAINTE QUI MORD VRAIMENT : LA FENETRE DE CALCUL ===")
    print("  Le parc doit etre rafraichi dans une fenetre nocturne. Debit requis")
    print("  = S / duree. Un worker a 8 vCPU sert 1000/latence inferences/s.")
    for S, win_min in ((20_000, 30), (500_000, 30), (500_000, 120)):
        deb = S / (win_min * 60)
        print(f"\n  S = {S:>7,} series, fenetre = {win_min:>3d} min "
              f"-> debit requis {deb:>7.1f} inf./s")
        print(f"{'modele':>16s} {'inf./s/worker':>14s} {'workers':>9s} "
              f"{'vCPU':>7s} {'x vs 5M':>9s}")
        base_w = None
        for nm, (pm, lat, mo) in MODELS.items():
            per = 1000.0 / lat
            wk = int(np.ceil(deb / per))
            base_w = base_w or wk
            print(f"{nm:>16s} {per:>14.1f} {wk:>9d} {wk*THREADS:>7d} "
                  f"{wk/base_w:>8.1f}x")
    print("\n  -> 209M contre 123M : +30 % de workers a provisionner pour un gain")
    print("     PUBLIE de 0,2 %. Sur cet axe, le 209M est domine sans discussion.")
    print("  -> et 5M contre 209M : un facteur 13 sur le parc materiel. C'est la")
    print("     que la taille se paie, pas sur la facture de vCPU-secondes.")
 
    print("\n=== D. LA GATE QUI BLOQUE, ET ELLE BLOQUE LES QUATRE TAILLES ===")
    print(f"  licence du DEPOT de code  : {CODE_LICENSE}  <- ce que montre le badge GitHub")
    print(f"  licence des POIDS         : {WEIGHTS_LICENSE}  <- distribuee ailleurs, "
          "sur le Hub")
    ALLOWED = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0"}
    for nm in MODELS:
        ok = WEIGHTS_LICENSE.lower() in ALLOWED
        print(f"  {nm:>16s} : deploiement commercial -> "
              f"{'AUTORISE' if ok else 'REFUSE (licence de poids non listee, '
                                       'texte non lu a la source)'}")
    print("\n  Le badge de licence d'un depot decrit le CODE. Les poids ont leur")
    print("  propre licence, publiee ailleurs. 'Modified MIT' n'est pas MIT :")
    print("  tant que son texte n'a pas ete lu, la gate doit refuser.")
    print("  Meme conclusion qu'en W42 pour TimesFM-3, obtenue par un chemin")
    print("  INVERSE : la ou TimesFM-3 affichait une licence restrictive,")
    print("  Timestral affiche un badge Apache-2.0 rassurant... sur le code.")
```
 
### Résultats obtenus
 
**A. Le coût CPU du serving, en euros par mois** (0,035 €/vCPU-h, 8 threads, 720 h/mois) :
 
| modèle | 20 000 séries, k=1 h | 500 000 séries, k=1 h | 500 000 séries, k=24 h |
|---|---|---|---|
| Timestral-5M (10 ms) | 11 € | 280 € | 12 € |
| Timestral-28M (28 ms) | 31 € | 784 € | 33 € |
| Timestral-123M (105 ms) | 118 € | 2 940 € | 123 € |
| Timestral-209M (140 ms) | 157 € | **3 920 €** | 163 € |
 
→ **Premier résultat, et il contredit l'intuition dominante : le coût CPU est négligeable.** Servir **500 000 séries,
rafraîchies toutes les heures, sur le plus gros modèle, coûte 3 920 € de CPU par mois.** Conséquence directe : le gain
d'exactitude nécessaire pour rembourser les **+33 %** de temps CPU du 209M sur le 123M n'est que de **0,327 %** à
k = 6 h pour un enjeu de 0,10 €/série/mois, et **le seuil de rentabilité tombe à 0,16 €/série/mois**. **Au-delà de 16
centimes d'enjeu par série et par mois, les 0,2 % publiés remboursent le surcoût.** Donc : **si vous choisissez le
petit modèle « pour économiser du calcul », vous optimisez la mauvaise variable.**
 
**B. La contrainte qui mord vraiment : la fenêtre de calcul.**
 
| S = 500 000 séries, fenêtre nocturne de 30 min → débit requis **277,8 inf./s** | inf./s/worker | workers | vCPU | × vs 5M |
|---|---|---|---|---|
| Timestral-5M | 100,0 | 3 | 24 | 1,0× |
| Timestral-28M | 35,7 | 8 | 64 | 2,7× |
| Timestral-123M | 9,5 | **30** | 240 | 10,0× |
| Timestral-209M | 7,1 | **39** | 312 | **13,0×** |
 
→ **Deuxième résultat : la taille se paie en parc matériel à provisionner, pas en vCPU-secondes facturées.** Le 209M
exige **+30 % de workers** par rapport au 123M **pour un gain publié de 0,2 %** : sur cet axe, **le 209M est dominé
sans discussion**. Et entre le 5M et le 209M, c'est un **facteur 13** sur le parc. **Le bon arbitrage de serving n'est
pas « combien coûte une inférence » mais « combien de workers pour tenir la fenêtre ».** Allonger la fenêtre de 30 à
120 minutes divise le parc par 4 — **le levier le moins cher de la liste est un créneau d'ordonnancement, pas un
modèle plus petit.**
 
**C. Et contre un modèle classique, dont le serving est quasi gratuit ?** Le modèle du bloc 2 (moyenne glissante +
profil saisonnier, en numpy) tourne en **0,08 ms**, soit **1 750× moins** que le 209M. À 1 €/série/mois d'enjeu, un
TSFM doit battre cette moyenne glissante de **0,13 %** (209M) ou **0,03 %** (28M) pour être rentable. **Autrement dit :
le coût de calcul ne protège pas le modèle classique.** Le seul argument qui le protège est celui de W42 — le
`n*` de break-even par série, et le fait qu'il n'a **aucune contrainte de licence**.
 
**D. La gate qui bloque — et elle bloque les quatre tailles.**
 
```
  licence du DEPOT de code  : Apache-2.0          <- ce que montre le badge GitHub
  licence des POIDS         : Modified MIT v1.0   <- distribuee ailleurs, sur le Hub
  Timestral-5M / 28M / 123M / 209M : deploiement commercial -> REFUSE
      (licence de poids non listee, texte non lu a la source)
```
 
### Quand cette approche est réellement supérieure — et quand elle ne l'est pas
 
**Quand un TSFM est le bon choix de serving :**
- **quand le parc est hétérogène et que le coût marginal d'une série nouvelle est nul.** Le zero-shot n'a pas de
  cold start : c'est son seul avantage structurel, et il est décisif sur un catalogue qui bouge.
- **quand la fenêtre de calcul est large** (nocturne de 2 h plutôt que 30 min) : le parc à provisionner divise par 4 et
  le 123M redevient banal.
- **quand l'enjeu unitaire dépasse ~0,16 €/série/mois**, seuil mesuré : au-delà, les 0,2 % de gain publié remboursent
  un modèle 33 % plus lent. Et à 10 €/série/mois, **toute considération de coût de calcul est du bruit**.
**Quand ce n'est pas le bon choix, et il faut le dire :**
- **quand on choisit la taille pour économiser du CPU.** 3 920 €/mois pour 500 000 séries à l'heure sur le plus gros
  modèle : ce n'est pas là qu'est l'argent. **Le vrai arbitrage est la fenêtre de calcul et la licence.**
- **quand la licence des poids n'a pas été lue à la source.** Trois familles, trois fois le même piège : le badge du
  dépôt décrit le code. **Et « Modified MIT » n'est pas MIT** : jusqu'à lecture, la gate refuse.
- **quand on compare le 209M au 123M sur un leaderboard.** 0,2 % d'écart publié, +33 % de latence, +30 % de workers.
  **Un classement qui met le 209M premier et le 123M deuxième est exact et inutilisable** — c'est la position de W42
  (« le leaderboard n'est pas un outil de décision ») confirmée par les chiffres du fournisseur lui-même.
- **quand le modèle classique est déjà au niveau.** À 1 €/série/mois d'enjeu, le TSFM doit le battre de 0,13 % : la
  barre est basse, donc **la question n'est pas le coût, c'est de savoir s'il le bat vraiment** — et ça se mesure avec
  le protocole de W40, pas avec un leaderboard.
---
 
## 7 — Stack Technique · 15 min
 
### Le serving de modèles en open source : l'implémentation de référence du protocole est à l'arrêt
 
| Critère | **`mlserver`** (Seldon) | **`bentoml`** |
|---|---|---|
| Version stable | **1.7.1** | **1.4.40** |
| Date de release | **6 juin 2025** — soit **16 mois** | **23 octobre 2026** — soit **2 jours** |
| Licence | Apache-2.0 | Apache-2.0 |
| Python supporté | **`>=3.9, <3.13`** | **`>=3.9`** |
| Cadence récente | `1.7.0` 11/04/2025, `1.7.1` 06/06/2025, puis **une seule pré-release `1.7.2rc1` le 16/12/2025 et plus rien** | 1.4.30 (27/11/2025) → 1.4.40 (23/10/2026) : **10 releases en 11 mois** |
| Rôle | implémentation de référence du **KServe V2 / Open Inference Protocol** | runtime de service généraliste, protocole propre |
| Sources | [pypi.org/project/mlserver](https://pypi.org/project/mlserver/) | [pypi.org/project/bentoml](https://pypi.org/project/bentoml/) |
 
**Le constat est factuel et il est désagréable.** `mlserver` est l'implémentation de référence du protocole d'inférence
que tout l'écosystème cite, et **sa dernière version stable a 16 mois et ne s'installe pas sur Python 3.13** — la
version de l'environnement de ce run (`3.13.17`). La borne `<3.13` n'est pas un oubli cosmétique : c'est une
incompatibilité déclarée, et elle signifie qu'un projet neuf doit **épingler Python 3.12** pour l'utiliser.
**Verdict du critère « encore maintenu » : échec.** Ce n'est pas une raison d'abandonner le protocole (`kserve`,
Triton, les serveurs propriétaires l'implémentent aussi) ; c'est une raison de **ne pas bâtir sur cette
implémentation-là**, et de le dire au client avant lui.
 
### Les 15 minutes, utiles
 
1. **5 min — vérifier la contrainte Python sur son propre socle.** `pip index versions mlserver` et la borne
   `requires-python` : si le projet est en 3.13, l'arbitrage est déjà tranché. Vérifier au passage que le runtime
   retenu (`bentoml`, `kserve`, Triton) expose bien les métadonnées de modèle du V2 — c'est le seul endroit où
   `model_version` est contractuel.
2. **5 min — écrire la couche `parameters` et la documenter comme non portable.** Puisque l'OIP n'a aucun champ
   temporel (bloc 3), lister explicitement les clés maison (`as_of`, `horizon`, `tau`, `training_asof`) dans un
   schéma versionné, et **écrire noir sur blanc dans le contrat que ces clés ne sont pas portables d'un serveur à
   l'autre.** C'est la ligne `protocol_cannot_express` du YAML.
3. **5 min — la clé de cache.** Vérifier que le cache de prévisions (CDN, Redis, couche applicative) inclut bien
   `as_of` **et** `training_asof`. Si la clé est `(series_id, horizon)`, deux appels séparés d'un mois rendent la
   même réponse et **personne ne verra jamais que le modèle n'a pas été réentraîné.** C'est le `−99,5 % silencieux`
   de W43, transposé : **un cache trop large est un bug qui rend le système plus calme.**
### Avis de mentor, sans ménagement
 
**La surcote de la semaine : « on sert nos modèles via l'Open Inference Protocol, donc c'est standardisé ».** Non.
Vérifié champ par champ : le protocole standardise **la forme des tenseurs**, et rien du temps. Pas d'`as_of`, pas
d'`horizon`, pas de niveaux de quantile, et une seule porte de sortie — l'objet `parameters`, dont la spécification
dit elle-même **« Currently no parameters are defined »**. **Un protocole qui ne peut pas exprimer la date à laquelle
une prévision a été faite n'est pas un standard de prévision : c'est un standard de tenseurs qu'on utilise pour
transporter des prévisions.** La conséquence pratique est immédiate et chère : **la clé de cache est hors contrat,
donc le comportement du cache est hors contrat.**
 
**La position à tenir, et c'est le prolongement exact de W43.** En W43, le MCP avait déplacé la fiabilité du transport
vers le contrat d'outil, et la leçon était : *« le sans-état n'est pas une amélioration de fiabilité, c'est un
transfert ; si vous ne l'avez pas écrit, elle a été perdue »*. **En W44 le constat est plus dur, parce qu'il n'y a même
pas eu de transfert : la sémantique temporelle d'une prévision n'a jamais été dans aucun protocole de serving. Elle
n'a pas été déplacée, elle n'a jamais été écrite.** Et l'ironie vaut d'être notée en entretien : **un protocole d'outils
d'agent publié en juillet 2026 impose `ttlMs` et `cacheScope` sur ses résultats, là où le protocole d'inférence de
modèles n'a toujours rien.**
 
---
 
## 8 — Inférence causale · 15 min
 
### Le problème : un réentraînement est une intervention, et le tableau de bord qui le mesure est biaisé par construction
 
Le mécanisme est d'une banalité totale et il est partout. On déclenche le réentraînement **quand l'erreur récente
dépasse un seuil**. On compare ensuite l'erreur **avant** et **après**. Mais on a **sélectionné sur une erreur élevée,
dont une partie est du bruit** : la régression vers la moyenne fait baisser l'erreur après l'intervention **même si le
réentraînement ne fait rien**.
 
```
   seuil franchi  <--- contient du BRUIT, pas seulement de la derive
         │
         ▼
   reentrainement  ────────────►  erreur APRES plus basse
         │                              ▲
         └──── regression vers la ──────┘
               moyenne (effet nul)
```
 
C'est le pendant exact du bloc 8 de W43 (où l'instrument dérivait avec l'agent), mais la cause est différente et plus
insidieuse : **ici l'instrument est parfait. C'est la règle de sélection qui fabrique l'effet.** Et c'est pourquoi
`DriftGuard` et consorts peuvent annoncer des ROI à trois chiffres sans mentir : le chiffre est réel, l'attribution ne
l'est pas.
 
### Code — **exécuté, sorties réelles**
 
```python
"""Bloc 8 - W44 : un reentrainement est une INTERVENTION, et le tableau de bord
qui le mesure est biaise par construction.
 
Le mecanisme. On declenche le reentrainement quand l'erreur recente depasse un
seuil. On compare ensuite l'erreur AVANT et APRES. Mais on a SELECTIONNE sur
une erreur elevee, dont une partie est du bruit : la regression vers la moyenne
fait baisser l'erreur apres l'intervention MEME SI LE REENTRAINEMENT NE FAIT
RIEN. C'est le "ca a marche" le plus repandu du MLOps.
 
Quatre estimateurs, et un placebo qui tranche :
  naif pre/post | DiD contre les series declenchees NON reentrainees |
  A/B randomise parmi les declenchees | placebo (reajustement a l'identique)
 
Dependances : numpy + scipy.
"""
from __future__ import annotations
 
import numpy as np
from scipy.stats import norm
 
L = 14            # longueur des fenetres avant / apres (2 semaines de
                  # suivi quotidien : c'est la fenetre des vrais tableaux de bord)
SIGMA = 1.0       # bruit d'erreur par pas
 
 
def simulate(n=20_000, thr=1.05, delta_true=0.30, seed=844,
             bias_scale=0.50, placebo=False, randomize=False, p_treat=0.5):
    """Chaque serie porte un biais de retard b_i >= 0 (l'erreur systematique du
    modele deploye). L'erreur observee est e = b_i + bruit.
 
    delta_true : part du biais que le reentrainement supprime REELLEMENT.
    placebo    : le reentrainement ne change rien (reajustement a l'identique).
    randomize  : parmi les series DECLENCHEES, on reentraine une fraction tiree
                 au hasard -> plan A/B, le seul estimateur sans hypothese.
    """
    rng = np.random.default_rng(seed)
    b = np.abs(rng.normal(0, bias_scale, n))
    pre = np.abs(b[:, None] + rng.normal(0, SIGMA, (n, L))).mean(axis=1)
    fired = pre > thr
    if randomize:
        treat = fired & (rng.random(n) < p_treat)
    else:
        treat = fired.copy()
    eff = 0.0 if placebo else delta_true
    b_post = np.where(treat, b * (1 - eff), b)
    post = np.abs(b_post[:, None] + rng.normal(0, SIGMA, (n, L))).mean(axis=1)
    return dict(b=b, pre=pre, post=post, fired=fired, treat=treat)
 
 
def truth(delta_true, bias_scale=0.50, n=400_000, seed=11, thr=1.05):
    """Effet causal vrai SUR LA POPULATION DECLENCHEE, mesure par assignation
    aleatoire sur un echantillon enorme."""
    d = simulate(n=n, thr=thr, delta_true=delta_true, seed=seed,
                 bias_scale=bias_scale, randomize=True, p_treat=0.5)
    t, c = d["treat"], d["fired"] & ~d["treat"]
    return float(d["post"][t].mean() - d["post"][c].mean())
 
 
def est_naive(d):
    t = d["treat"]
    return float(d["post"][t].mean() - d["pre"][t].mean())
 
 
def est_did(d):
    """DiD : les series qui ont FRANCHI le seuil mais n'ont pas ete reentrainees
    servent de controle. Valide seulement si l'assignation est ignorable."""
    t = d["treat"]; c = d["fired"] & ~d["treat"]
    if c.sum() == 0:
        return float("nan")
    return float((d["post"][t] - d["pre"][t]).mean()
                 - (d["post"][c] - d["pre"][c]).mean())
 
 
def est_ab(d):
    t = d["treat"]; c = d["fired"] & ~d["treat"]
    if c.sum() == 0:
        return float("nan")
    return float(d["post"][t].mean() - d["post"][c].mean())
 
 
def ci_ab(d, alpha=0.05):
    t = d["treat"]; c = d["fired"] & ~d["treat"]
    a, bb = d["post"][t], d["post"][c]
    se = np.sqrt(a.var(ddof=1) / a.size + bb.var(ddof=1) / bb.size)
    z = norm.ppf(1 - alpha / 2)
    diff = a.mean() - bb.mean()
    return float(diff - z * se), float(diff + z * se)
 
 
if __name__ == "__main__":
    DELTA = 0.30
    tv = truth(DELTA)
    print(f"effet causal VRAI du reentrainement sur l'erreur moyenne des series")
    print(f"declenchees (plan randomise, 400 000 series) = {tv:+.4f}\n")
 
    print("1) le tableau de bord standard : on reentraine TOUT ce qui franchit "
          "le seuil")
    d = simulate(randomize=False, delta_true=DELTA)
    print(f"   series declenchees : {d['fired'].mean():.1%} du parc")
    print(f"   estimateur naif pre/post : {est_naive(d):+.4f}  "
          f"(biais {est_naive(d)-tv:+.4f}, soit "
          f"{abs(est_naive(d)-tv)/abs(tv):.0%} de l'effet reel)")
    print("   DiD et A/B : INCALCULABLES -- il n'y a aucun controle, on a tout "
          "reentraine.")
 
    print("\n2) LE PLACEBO : on 'reentraine' a l'identique (effet vrai = 0)")
    dp = simulate(randomize=False, placebo=True)
    print(f"   estimateur naif pre/post : {est_naive(dp):+.4f}  <-- "
          f"{abs(est_naive(dp))/abs(tv):.0%} de l'effet reel, "
          f"a partir de ZERO changement")
    print("   C'est de la pure regression vers la moyenne : on a selectionne")
    print("   sur une erreur elevee, dont une partie etait du bruit.")
 
    print("\n3) l'effet du SEUIL : plus on est selectif, plus le faux effet grossit")
    print(f"{'seuil':>8s} {'% declenche':>12s} {'naif (placebo)':>16s} "
          f"{'% de l effet reel':>18s}")
    for thr in (0.80, 0.90, 1.00, 1.05, 1.15, 1.30, 1.50):
        dd = simulate(thr=thr, randomize=False, placebo=True)
        if dd["treat"].sum() < 50:
            continue
        print(f"{thr:>8.2f} {dd['fired'].mean():>11.1%} "
              f"{est_naive(dd):>16.4f} {abs(est_naive(dd))/abs(tv):>17.0%}")
 
    print("\n4) le plan qui marche : A/B parmi les series DECLENCHEES")
    da = simulate(randomize=True, delta_true=DELTA, p_treat=0.5)
    nt, nc = int(da["treat"].sum()), int((da["fired"] & ~da["treat"]).sum())
    lo, hi = ci_ab(da)
    print(f"   {nt} reentrainees / {nc} temoins (toutes ont franchi le seuil)")
    print(f"   naif pre/post : {est_naive(da):+.4f}  (biais {est_naive(da)-tv:+.4f})")
    print(f"   DiD           : {est_did(da):+.4f}  (biais {est_did(da)-tv:+.4f})")
    print(f"   A/B randomise : {est_ab(da):+.4f}  (biais {est_ab(da)-tv:+.4f})  "
          f"IC 95 % [{lo:+.4f} ; {hi:+.4f}]")
 
    print("\n5) et le placebo du plan A/B : il doit rendre ZERO")
    dap = simulate(randomize=True, placebo=True, p_treat=0.5)
    lo2, hi2 = ci_ab(dap)
    print(f"   naif pre/post : {est_naive(dap):+.4f}  <-- toujours faux")
    print(f"   DiD           : {est_did(dap):+.4f}")
    print(f"   A/B randomise : {est_ab(dap):+.4f}  IC 95 % [{lo2:+.4f} ; {hi2:+.4f}]  "
          f"-> {'PASSE' if lo2 <= 0 <= hi2 else 'ECHOUE'} le placebo")
 
    print("\n6) le cout d'un temoin : quelle fraction laisser NON reentrainee ?")
    print(f"{'p_traite':>10s} {'temoins':>9s} {'A/B':>9s} {'demi-largeur IC':>17s} "
          f"{'perte de service':>17s}")
    for p in (0.5, 0.7, 0.9, 0.95, 0.99):
        dz = simulate(randomize=True, delta_true=DELTA, p_treat=p)
        lo3, hi3 = ci_ab(dz)
        nc3 = int((dz["fired"] & ~dz["treat"]).sum())
        # perte de service = erreur supplementaire portee par les temoins
        loss = (dz["post"][dz["fired"] & ~dz["treat"]].mean()
                - dz["post"][dz["treat"]].mean()) * (1 - p)
        print(f"{p:>10.2f} {nc3:>9d} {est_ab(dz):>+9.4f} {(hi3-lo3)/2:>17.4f} "
              f"{loss:>17.4f}")
    print("   -> a p = 0,95, l'IC reste exploitable et le cout de service est")
    print("      le 5 % de series laissees en retard. Un temoin permanent de")
    print("      quelques pour cent est le prix d'un chiffre qui veut dire quelque chose.")
```
 
**Sorties réelles :**
 
```
effet causal VRAI du reentrainement sur l'erreur moyenne des series
declenchees (plan randomise, 400 000 series) = -0.1053
 
1) le tableau de bord standard : on reentraine TOUT ce qui franchit le seuil
   series declenchees : 20.4% du parc
   estimateur naif pre/post : -0.3119  (biais -0.2066, soit 196% de l'effet reel)
   DiD et A/B : INCALCULABLES -- il n'y a aucun controle, on a tout reentraine.
 
2) LE PLACEBO : on 'reentraine' a l'identique (effet vrai = 0)
   estimateur naif pre/post : -0.2072  <-- 197% de l'effet reel,
                                           a partir de ZERO changement
 
3) l'effet du SEUIL : plus on est selectif, plus le faux effet grossit
   seuil  % declenche   naif (placebo)  % de l effet reel
    0.80       63.7%          -0.0890                85%
    0.90       43.9%          -0.1351               128%
    1.00       26.9%          -0.1830               174%
    1.05       20.4%          -0.2072               197%
    1.15       11.5%          -0.2389               227%
    1.30        4.5%          -0.2745               261%
    1.50        1.3%          -0.2993               284%
 
4) le plan qui marche : A/B parmi les series DECLENCHEES
   2080 reentrainees / 2007 temoins (toutes ont franchi le seuil)
   naif pre/post : -0.3033  (biais -0.1980)
   DiD           : -0.0992  (biais +0.0061)
   A/B randomise : -0.0993  (biais +0.0060)  IC 95 % [-0.1141 ; -0.0844]
 
5) et le placebo du plan A/B : il doit rendre ZERO
   naif pre/post : -0.1974  <-- toujours faux
   DiD           : +0.0067
   A/B randomise : +0.0066  IC 95 % [-0.0102 ; +0.0234]  -> PASSE le placebo
 
6) le cout d'un temoin : quelle fraction laisser NON reentrainee ?
  p_traite   temoins       A/B   demi-largeur IC  perte de service
      0.50      2007   -0.0993            0.0148            0.0496
      0.70      1188   -0.1063            0.0171            0.0319
      0.90       389   -0.1045            0.0275            0.0104
      0.95       187   -0.1144            0.0352            0.0057
      0.99        46   -0.1079            0.0728            0.0011
```
 
### Les cinq lectures qui comptent
 
1. **Le tableau de bord standard triple l'effet.** Effet causal vrai **−0,1053**, estimateur naïf pré/post
   **−0,3119** : un **biais de 196 % de l'effet réel**. L'équipe qui publie « le réentraînement a divisé notre erreur
   par trois » dit vrai sur le chiffre et faux sur la cause.
2. **Le placebo chiffre exactement la part fabriquée : −0,2072, soit 197 % de l'effet réel, à partir de zéro
   changement.** Donc **66 % de l'amélioration affichée (0,2072 / 0,3119) est de la pure régression vers la moyenne.**
   On « réajuste à l'identique » et le tableau de bord félicite l'équipe.
3. **Et c'est le seuil prudent qui fabrique le plus gros faux effet — l'inverse de l'intuition.** De **85 %** de
   l'effet réel quand on déclenche sur 63,7 % du parc à **284 %** quand on ne déclenche que sur 1,3 %. Raison :
   plus le seuil est haut, plus la part de bruit dans le franchissement est grande, donc plus la régression vers la
   moyenne est violente. **« On ne réentraîne que quand c'est vraiment nécessaire » est la configuration qui produit
   le rapport le plus flatteur et le moins vrai.**
4. **Et le piège structurel : la politique standard détruit sa propre capacité de mesure.** Si l'on réentraîne
   **100 %** de ce qui déclenche, **le DiD et l'A/B sont incalculables** — il n'y a aucun témoin. Ce n'est pas une
   négligence statistique, c'est une **conséquence de l'architecture de la politique.** Le correctif n'est pas un
   meilleur estimateur, c'est une **ligne de contrat** : `holdout.permanent: true`.
5. **Le prix d'un chiffre qui veut dire quelque chose : 5 % du parc.** À `p_traite = 0,95`, l'A/B rend **−0,1144**
   avec un IC de demi-largeur **0,0352** (contre un effet de 0,1053 : largement résolu) pour **187 témoins** et une
   perte de service de **0,0057** d'erreur moyenne sur le parc. À `p = 0,99`, l'IC double (0,0728) et devient
   inutilisable. **La fraction témoin se dimensionne, elle ne se devine pas** — et 5 % est le bon ordre de grandeur.
**Les conséquences de contrat**, toutes trois dans `forecast_service.yaml` :
`retraining.holdout.{permanent: true, fraction: 0.05}` · `retraining.forbid_drift_recall_as_kpi: true` ·
`observability.signals[4]: effet du réentraînement mesuré contre le témoin`.
Et une règle de conduite que je facture : **avant de croire qu'un réentraînement a servi, rejouez-le à l'identique sur
un échantillon. Si le placebo rend la moitié de votre gain, votre gain est de la régression vers la moyenne.**
Même structure que les placebos de W40, W42 et W43 : **un estimateur qui n'a pas passé un placebo n'est pas un
estimateur, c'est un chiffre.**
 
---
 
## 9 — Positionnement freelance & capitalisation · 10 min
 
### Article de blog — 10ᵉ épisode du *Carnet d'un Time Series Engineer*
 
**Titre :** « Réentraîner n'est pas une hygiène, c'est une décision »
**Angle :** la question « à quelle cadence réentraîner ? » n'a pas une réponse mais trois, incompatibles, séparées par
deux ordres de grandeur — et les deux instruments qu'on utilise pour y répondre (le détecteur de dérive, le tableau de
bord avant/après) sont tous les deux faux, de façons mesurables. Pas un billet d'opinion sur le MLOps : trois mesures
et un contrat de service.
 
#### Front matter Hugo (Blowfish)
 
```yaml
---
title: "Réentraîner n'est pas une hygiène, c'est une décision"
date: 2026-11-01T09:00:00+01:00
draft: false
description: "Un détecteur de dérive parfait perd contre un cron hebdomadaire. 66 % de l'amélioration qu'affiche un tableau de bord de réentraînement est de la régression vers la moyenne. Et le budget de fraîcheur d'un service de prévision varie d'un facteur 7 selon le quantile servi."
summary: "Trois contraintes incompatibles sur la même cadence : k=1 pour la stabilité, k=24 pour le quantile 0,95, k=336 pour l'économie. Avec les formes closes du churn, la gate de licence des poids, et pourquoi l'Open Inference Protocol ne peut pas exprimer une prévision."
categories: ["Time Series"]
tags:
  - "MLOps"
  - "serving"
  - "réentraînement"
  - "drift"
  - "stabilité de prévision"
  - "SLA"
  - "foundation models"
  - "Timestral"
  - "inférence causale"
series: ["Carnet d'un Time Series Engineer"]
series_order: 10
showTableOfContents: true
showHero: true
heroStyle: "background"
---
```
 
> **Note de taxonomie.** Retour à `categories: ["Time Series"]` après l'épisode 9 en `["Agentic AI"]`. La `series` est
> la même : c'est le même ingénieur. Le filtre par catégorie permet au lecteur Head of Data de suivre la ligne
> « séries temporelles » sans trier, tout en voyant que l'auteur sort du périmètre quand il le faut.
 
#### Trame en 9 sections
 
| § | Section | Contenu | Matière |
|---|---|---|---|
| 1 | **L'anecdote** | « Nous avions un détecteur de dérive parfait. Il a réentraîné 3,7 fois par série, toujours au bon moment, et il a perdu contre un cron hebdomadaire. » | bloc 2 |
| 2 | **Trois contrats, une seule cadence** | les trois `k_max` incompatibles : 1, 24, 336 | bloc 5 |
| 3 | **Le churn ne disparaît pas, il se concentre** | ×223,6 sur le saut, ÷3,2 sur le churn/pas ; la part de bruit à 67,6 % | bloc 4 |
| 4 | **La fenêtre n'est pas la cadence** | `w* = 27` identique à `k = 1, 24, 168` ; et le contre-exemple de tendance | bloc 4 |
| 5 | **Le quantile décide du budget de fraîcheur** | +97 % contre +41 % ; `k_max` 24 contre 168 | bloc 5 |
| 6 | **L'illusion de la latence de clôture** | +6,3 % en MASE, **+12,5 %** au quantile servi | blocs 2 et 5 |
| 7 | **Le protocole ne sait pas dire *quand*** | l'OIP V2 champ par champ : aucun `as_of`, aucun horizon, aucun quantile | bloc 3 + spec |
| 8 | **La licence des poids n'est pas celle du dépôt** | Timestral et Chronos-2, le même piège, trois familles | bloc 1 + bloc 6 |
| 9 | **Prouvez que le réentraînement a servi** | placebo à 197 %, le seuil prudent qui ment le plus, le témoin à 5 % | bloc 8 |
 
**Figures à produire (3, pas plus) :** (1) le plan `saut p95 × fréquence des sauts` avec les politiques du bloc 2
placées dessus et la zone des clauses signables hachurée — **la figure qui montre l'incompatibilité** ; (2) `part de
bruit dans la variance publiée` en fonction de `k`, avec les deux régimes annotés ; (3) le faux effet du placebo en
fonction du taux de déclenchement — **la courbe qui monte quand on devient prudent**.
 
**Positionnement.** Cette semaine est **la plus directement vendable des cinq dernières**, parce qu'elle ne parle pas
de modèles : elle parle de **ce qu'un service de prévision promet par écrit**. Et elle boucle la série d'audits :
W40 (protocole d'évaluation), W41 (métrique de décision), W42 (déployabilité du modèle), W43 (contrat d'outil),
**W44 (contrat de service)**. **Cinq audits, une seule promesse : vous ne saurez pas si ça marche tant que quelqu'un
n'aura pas écrit ce que « marcher » veut dire — et cette semaine, ce qu'il faut écrire a trois clauses qui se
contredisent, ce qui est exactement la raison pour laquelle personne ne l'écrit.**
 
---
 
## ✅ Clôture
 
### Tableau récapitulatif
 
| Bloc | Durée | Livrable produit |
|---|---|---|
| 1 — Veille | 20 min | 4 ressources vérifiées à la source : **Timestral (15/10/2026)** — 4 tailles, latences CPU publiées, et **le piège de licence : dépôt Apache-2.0, poids en « Modified MIT v1.0 » distribués ailleurs** · arXiv 2604.06438 (dette d'apprentissage KL : **72/72** contre le calendaire, **58/72** contre CUSUM, objectifs relatifs **0,677 / 0,975**) · arXiv 2506.05776 + 2505.00356 (**SMQC**, stabilité, 10 modèles) · arXiv 2609.39789 (**PILOT**, pseudo-étiquette future / déclencheur à erreurs closes). **Dette W42 caractérisée** : `amazon-science/chronos-forecasting` est Apache-2.0 **et ne dit rien des poids** — même patron. **1 ressource écartée** (DriftGuard : « ROI jusqu'à 417 » **sans unité**, classé cs.CV). |
| 2 — Ingénierie & code | 35 min | `retrain_policy_harness.py` **exécuté**, 10 politiques : **détecteur de ruptures PARFAIT à 100 % de réentraînements utiles et MASE 1,3685 contre 0,9829 pour un cron hebdo** · **saut p95 ×222 / fréquence ÷126 du continu au mensuel** · **latence de clôture = +6,3 % d'illusion de MASE** · la politique optimale change de famille avec λ_k (CUSUM gagne à λ_k ≥ 10) |
| 3 — System design & MLOps | 20 min | **L'Open Inference Protocol V2 vérifié champ par champ : aucun `as_of`, aucun horizon, aucun niveau de quantile, et « Currently no parameters are defined »** → architecture à 4 étages (gate à 6 règles bloquantes → plan de données bitemporel → passerelle qui porte la sémantique temporelle → 4 signaux dont 3 inexistants ailleurs) + `forecast_service.yaml` avec `freshness.budget_by_tau` chiffré, `holdout.permanent`, `repo_license_is_not_weights_license` + 3 arbitrages tranchés |
| 4 — Fondamentaux | 20 min | **Formes closes validées MC à ≤ 1,8 % sur 8 configurations** : `MSE(k,w) = σ²(1+1/w) + q[(ā+1) + (w−1)(2w−1)/(6w)]` · `w* = √(3(σ²+q/6)/q)` **indépendant de `k`** (argmin = 27 pour k=1, 24 et 168) · **décomposition de `Var(R)` en jitter d'estimation et dérive réelle** → **67,6 % de bruit publié à k=1, 0,3 % à k=720** · deux régimes de churn (`1/√k` puis constante) · **contre-exemple de tendance : la cadence retire 75 % du biais et 100 % du reste est le biais de fenêtre** · 1 question d'entretien + réponse en 3 points chiffrés |
| 5 — Mini-projet | 40 min | `freshness_cost.py` **exécuté** sur 200 séries : **la triple contrainte incompatible — `k=1` (stabilité à 10 %), `k=24` (τ=0,95 contre la naïve), `k=336` (optimum économique à 2 €)** · **`k_max` de 168 à 24 selon τ : facteur 7** · dégradation **+41 % (τ=0,50) contre +97 % (τ=0,95)** · **à k=720 le modèle est 91,6 % PIRE que `y_{t−24}` au quantile servi** · illusion de latence **+12,5 % au quantile** contre +6,3 % en MASE · clause de stabilité chiffrée (saut p95 de 2,7 % à 305,5 %) |
| 6 — Serving des TSFM | 20 min | `serving_breakeven.py` **exécuté** sur les seules latences publiées : **le coût CPU est négligeable (3 920 €/mois pour 500 000 séries à l'heure sur le 209M)**, donc **le seuil de rentabilité du 209M n'est que de 0,16 €/série/mois** · **la contrainte qui mord est la fenêtre de calcul : 39 workers contre 30 (+30 %) pour un gain publié de 0,2 % → le 209M est dominé** · facteur **13** entre 5M et 209M sur le parc matériel · **gate de licence : les 4 tailles REFUSÉES** |
| 7 — Stack technique | 15 min | Fiche comparée : **`mlserver` 1.7.1 du 06/06/2025 (16 mois), Python `<3.13` → échoue au critère « encore maintenu »**, alors que c'est l'implémentation de référence de l'OIP · **`bentoml` 1.4.40 du 23/10/2026**, Apache-2.0, py ≥ 3.9, 10 releases en 11 mois · 3 vérifications de 5 min dont la clé de cache · position « un protocole qui ne sait pas dire *quand* n'est pas un standard de prévision » |
| 8 — Inférence causale | 15 min | `retrain_causal.py` **exécuté** : effet vrai **−0,1053**, naïf **−0,3119 (biais 196 %)**, **placebo −0,2072 = 197 % de l'effet à partir de zéro changement ⇒ 66 % du gain affiché est de la régression vers la moyenne** · **le faux effet croît de 85 % à 284 % quand le seuil devient prudent** · A/B randomisé à **+0,0060** de biais et **passe le placebo** · **témoin à 5 % : IC demi-largeur 0,0352 pour 0,0057 de perte de service** · et le piège structurel : **réentraîner 100 % des déclenchements rend l'effet incalculable** |
| 9 — Positionnement | 10 min | Front matter Hugo/Blowfish (`categories: ["Time Series"]`, `series_order: 10`) + trame en 9 sections + 3 figures spécifiées + note de taxonomie + bouclage de la série des 5 audits |
| **Total** | **195 min** | **Aucun élagage nécessaire** |
 
### Élément d'offre commerciale
 
> **Mission type — « Audit de cadence de réentraînement & contrat de service de prévision »**
> *5 jours.* Livrables :
> 1. **La triple contrainte chiffrée sur le parc du client** — trois cadences maximales calculées séparément
>    (exactitude **par quantile servi**, clause de stabilité, optimum économique), avec le tableau en euros et le
>    prix de calcul qui fait basculer la décision. Sur mon banc elles sont séparées par **deux ordres de grandeur** ;
>    chez le client, le livrable est de savoir **laquelle son ordonnanceur a choisie à sa place**.
> 2. **La mesure de l'illusion de latence de clôture** — le backtest rejoué avec la vraie fenêtre de données.
>    Mesuré : **+6,3 % en MASE, +12,5 % au quantile 0,95.** C'est la seule partie de la mission dont le résultat est
>    toujours négatif pour le client, et c'est celle qu'il paie le plus volontiers.
> 3. **`forecast_service.yaml` versionné** — budget de fraîcheur indexé sur le quantile, clause de stabilité
>    (saut p95 **et** fréquence des sauts visibles), gate de licence des poids, latence de clôture déclarée,
>    `pinned_together`, témoin permanent.
> 4. **Le dispositif de mesure causale du réentraînement** — témoin permanent dimensionné (5 % sur mon banc : IC
>    exploitable pour 0,0057 de perte de service), A/B parmi les séries déclenchées, et **le placebo obligatoire**.
> 5. **La note de provisionnement** — si un foundation model est envisagé : workers nécessaires par taille pour tenir
>    la fenêtre de calcul, et le gain d'exactitude qu'il faudrait pour justifier chaque palier.
>
> **TJM : 1 000 – 1 200 €**, ou **forfait 6 000 € HT**.
>
> **Question de qualification :** *« Votre SLA de fraîcheur, c'est combien de temps — et il vaut pour quel
> quantile ? »*
> - *« Une semaine »* → demander lequel. S'il n'y en a qu'un, montrer le facteur **7** : `k_max = 168` pour la
>   médiane, **24** pour le quantile 0,95. Leur SLA est juste pour la moitié de leur service.
> - *« On réentraîne quand le monitoring de drift alerte »* → **la meilleure entrée en mission.** Leur montrer qu'un
>   détecteur **parfait** fait 100 % de réentraînements utiles et **perd contre un cron hebdomadaire** (1,3685 contre
>   0,9829), parce que la dérive lente n'est pas une rupture. Puis demander de combien leur erreur a baissé, **mesuré
>   contre un témoin** — et s'il n'y a pas de témoin, montrer le placebo à **197 %**.
> - *« On réentraîne tous les mois, ça suffit »* → leur montrer qu'au quantile de service leur modèle est **91,6 %
>   pire que `y_{t−24}`**. Une naïve saisonnière les battrait. C'est l'argument le plus court de tout mon catalogue.
> - *« Nos modèles sont servis via l'Open Inference Protocol, c'est standardisé »* → ouvrir la spécification avec eux.
>   **Aucun `as_of`, aucun horizon, aucun niveau de quantile.** Donc leur clé de cache est hors contrat — et un cache
>   trop large masque un modèle qui n'a pas été réentraîné depuis six mois.
 
### Rétrospective — en une phrase vendable
 
**Je sais écrire le contrat de service d'un parc de prévisions — budget de fraîcheur indexé sur le quantile servi,
clause de stabilité chiffrée en amplitude et en fréquence, latence de clôture déclarée et appliquée au backtest, gate
de licence des poids, témoin permanent dimensionné — et je sais démontrer, chiffres en main, que ni le détecteur de
dérive ni le tableau de bord avant/après ne disent ce que le client croit qu'ils disent.**
 
### Graine de la semaine prochaine
 
**W45** — run du **dimanche 01/11/2026** (semaine ISO **44**) : `44 mod 10 = 4` → **Time Series** · axe
`44 mod 8 = 4` → **index 4 = causalité & uplift**. **Jamais joué en axe principal** : croisé en W40 (placebos de
backtest), W41 (spirale de demande censurée), W42 (`covariate_audit.py`, faux levier à +1,0786), W43 (dérive du
correcteur, placebo à 101 %) et **W44 (régression vers la moyenne du réentraînement, placebo à 197 %)** — cinq
croisements, aucun passage en principal. **Il est temps.**
 
**Angle posé ce run :** *l'intervention sur une série temporelle, et pourquoi presque tous les effets mesurés en
production sont des artefacts de sélection.* Le pont est déjà construit : le bloc 8 de cette semaine a montré que
**la règle de déclenchement fabrique l'effet**, et que **le faux effet croît avec la prudence du seuil** (85 % → 284 %).
C'est le même mécanisme que celui qui fait qu'une promotion « marche » parce qu'on la déclenche sur un creux de
demande. Pistes :
- **contrôle synthétique et causal impact** sur une intervention datée (promotion, changement tarifaire, panne), avec
  le placebo spatial **et** le placebo temporel ;
- **uplift sous demande censurée** : croiser la spirale de W41 (τ = 0,80 → dérive naïve −18,2 % contre EM −5,4 %) avec
  l'estimation d'un effet de traitement — **un traitement qui augmente la demande augmente aussi la censure** ;
- **la régression vers la moyenne comme confondant générique des interventions déclenchées par un seuil**, formalisée
  et chiffrée (ce run l'a mesurée, pas dérivée) ;
- **anticipation et effets de report** : une intervention sur une série temporelle n'a pas un effet, elle a un profil
  d'effets avec un pré-trend — et le pré-trend est exactement ce que le déclencheur par seuil garantit.
**Dettes ouvertes, par ordre de priorité :**
1. **Licences des poids des TSFM — `huggingface.co` bloqué par le proxy depuis W42.** La dette est désormais
   **caractérisée** (le patron est général : dépôt Apache-2.0, poids ailleurs) mais **aucun texte de licence n'a été
   lu à la source**. Concerne `amazon/chronos-2`, `mistralai/Timestral-*`. **Pistes non encore essayées :** le dépôt
   `mistralai/timestral` pourrait ajouter un `LICENSE.weights` ; la page `amazon.science` ; les notes de version
   d'`autogluon.timeseries`, qui embarque Chronos et doit donc documenter la licence des poids qu'il télécharge.
2. **Téléchargement Zenodo bloqué (403)** — découvert ce run. La page se lit, le fichier non. **Tous les
   mini-projets restent donc sur DGP calibré.** Piste : un miroir des jeux Monash sur un domaine autorisé, ou
   `forecastingdata.org`.
3. **Les chiffres des deux papiers Zanotti (2505.00356, 2506.05776) ne sont pas vérifiés** : les abstracts n'en
   contiennent aucun. Lire `arxiv.org/html/2506.05776v3` pour récupérer les valeurs de SMQC et les écarts
   d'exactitude par calendrier — **c'est la seule source qui pourrait valider empiriquement la forme close du
   bloc 4.**
> **Rappel de la règle de méthode de W43, et elle a tenu ce run :** une note de priorité d'axe écrite à l'avance est
> inopposable à la règle déterministe. L'index 4 (causalité & uplift) ci-dessus **n'est pas une promesse** : c'est le
> résultat du calcul `44 mod 8 = 4`, à **recalculer au run**. Et l'axe **index 5 (pricing / demand / anomaly
> detection)** reste **le seul jamais joué en principal et sans date** : il exige une semaine Time Series avec
> `W mod 8 = 5`, soit `W = 53` ou `W = 61` — **pas avant 2027**. Ne pas le promettre.
 
---
 
## Annexe — traçabilité de ce plan
 
- **Environnement d'exécution du run (25/10/2026) :** Python **3.13.17**, numpy **2.5.3**, scipy **1.18.1**,
  pandas **3.0.5**, scikit-learn **1.9.1**, jsonschema **4.26.0**, pydantic **2.13.5**, `referencing` présent,
  matplotlib **3.11.2**, PyYAML **6.0.3**, requests **2.34.2**. **`torch`, `statsmodels`, `river` ABSENTS et non
  installés.** Conséquence tenue : tout le code de ce plan tourne en **numpy + scipy**, sans réseau, sans clé API,
  sans GPU.
- **Code exécuté avant livraison :** **5 scripts, 5/5 exécutions réussies** — `retrain_policy_harness.py`,
  `churn_theory.py`, `freshness_cost.py`, `serving_breakeven.py`, `retrain_causal.py`.
  **Tous les chiffres de ce plan sont des sorties réelles**, reconfrontées ligne à ligne aux sorties.
- **Trois erreurs corrigées en cours de run, consignées parce qu'elles sont instructives :**
  1. **Les formes closes du bloc 4 étaient fausses d'un facteur 17 sur `Var(R)`** à la première écriture (terme de
     dérive posé à `q·k` au lieu du comptage exact des multiplicités). **Détectée uniquement par la validation
     Monte-Carlo.** Règle : *une forme close non confrontée à un Monte-Carlo n'est pas un résultat, c'est une
     conjecture.*
  2. **Le premier piège du bloc 2 était un faux piège.** Un CUSUM sur l'erreur à horizon `h` décalée de `h` pas est
     statistiquement identique au même CUSUM non décalé : la fuite mesurait **+0,01 %**, c'est-à-dire rien. Remplacé
     par la **latence de clôture des données**, qui est la vraie fuite de serving et vaut **+6,3 %**. Règle : *une
     fuite temporelle qui ne change pas le résultat n'est pas une fuite, c'est un décalage.*
  3. **L'écart-type des quantiles du bloc 5 incluait le profil saisonnier**, ce qui faisait perdre le modèle contre
     la naïve saisonnière de **−41 %** à τ = 0,95. Corrigé en retirant le profil du résidu. Règle : *un modèle qui
     perd contre `y_{t−24}` au quantile haut a un bug de dispersion avant d'avoir un problème de cadence.*
- **URL vérifiées par récupération directe le 25/10/2026 (13 liens, 0 lien mort) :**
  `mistral.ai/news/timestral/` · `docs.mistral.ai/models/timestral-1-0-209m` · `github.com/mistralai/timestral` ·
  `github.com/amazon-science/chronos-forecasting` · `arxiv.org/abs/2604.06438` · `arxiv.org/abs/2609.39789` ·
  `arxiv.org/abs/2506.05776` · `arxiv.org/abs/2505.00356` · `arxiv.org/abs/2601.08928` ·
  `zenodo.org/records/4656140` · `pypi.org/project/mlserver/` · `pypi.org/project/bentoml/` ·
  `docs.seldon.ai/seldon-core-2/user-guide/v2`.
- **Ressource écartée après vérification :** arXiv 2601.08928 (DriftGuard) — **« ROI jusqu'à 417 » sans unité**,
  classement **cs.CV** pour un sujet de chaîne logistique, corpus M5 de compétition. Règle de W43 reconduite : *une
  revendication chiffrée sans unité n'est pas un chiffre.* Gardée comme piste.
- **Ressources retenues pour leur cadre et non pour leurs chiffres :** arXiv 2506.05776 et 2505.00356 — **les deux
  abstracts ne contiennent aucune valeur numérique**. Citées pour le **SMQC** et pour le dispositif, pas pour des
  résultats non vérifiés. Dette consignée ci-dessus.
- **Réflexes de session reconduits (W40 → W43) :** `huggingface.co` **bloqué** · `pypi.org/project/<pkg>/` donne
  version, date, licence et bornes Python d'un coup (utilisé 3 fois : `mlserver`, `bentoml`, et les vérifications
  croisées) · `arxiv.org/abs/<id>` a suffi pour les 5 papiers · **4 recherches web en parallèle passent, 2 WebFetch
  en parallèle passent** (reconfirmé 6 fois ce run) · `gh api` sur dépôt tiers échoue en 403 → WebFetch sur la page.
- **Nouvelle contrainte réseau découverte ce run :** **`zenodo.org/records/<id>/files/<nom>.zip` est bloqué par le
  proxy (`CONNECT tunnel failed, 403`)** alors que `zenodo.org/records/<id>` se lit par WebFetch. **La page d'un jeu
  de données est vérifiable, son contenu n'est pas téléchargeable.** À consigner définitivement : les mini-projets de
  ce cadre restent sur DGP calibré, et le parseur `.tsf` de W41 est l'adaptateur à livrer au client.
- **Pièges numpy / méthode reconduits :** `np.diff(a, n=m)` est la différenciation d'**ordre** `m`, pas le retard `m`
  — le dénominateur saisonnier de la MASE s'écrit `a[m:] - a[:-m]` à la main (W42) · un bug qui rend le modèle
  *meilleur* ou le système *plus calme* ne lève aucune alerte (W42, W43) — et cette semaine ajoute le troisième
  membre de la famille : **un bug qui rend le système plus STABLE non plus** (le churn moyen d'un modèle gelé est
  exactement nul, ce qui fait conclure que ne jamais réentraîner est la politique la plus stable).
- **Matière collectée non utilisée (W44, à resservir) :** arXiv 2412.08435 (*Proactive Model Adaptation Against
  Concept Drift for Online Time Series Forecasting*) · arXiv 2506.14790 (*Continuous Evolution Pool*, dérive de
  concept récurrente) · arXiv 2510.14814 (*Tackling Time-Series Forecasting Generalization via Mitigating Concept
  Drift*) · arXiv 2605.24381 (*Assessing the Operational Viability of Foundation Models for Time Series
  Forecasting* — déjà utilisé en W39 pour le *throughput gap* 1000×, **à relire sous l'angle de la fenêtre de
  calcul du bloc 6**) · le *Decoupled Horizon Forecasting* de Timestral, revendiqué dans l'annonce et **non décrit
  techniquement** : chercher s'il existe un rapport technique.
- **Matière des semaines précédentes toujours en réserve :** W43 — arXiv 2608.13959 (*Repair, Not Improvement*,
  abstention d'appel d'outil, **non lu**) · arXiv 2605.25998 · arXiv 2610.14381 (ACED-Bench) · arXiv 2606.05558
  (off-policy evaluation par modèles de monde, **pertinent pour l'axe causal de W45**) · W42 — arXiv 2605.26161
  (**TSFMAudit**, contamination de données dans les TSFM, **non lu** — et désormais très pertinent face à Timestral) ·
  arXiv 2505.23195 (*Less is More*, élagage structuré de TSFM — **le complément direct du bloc 6**) · W41 —
  arXiv 2203.10716 (*Forecast Evaluation for Data Scientists*) · arXiv 2505.16319v3 (FreshRetailNet).
