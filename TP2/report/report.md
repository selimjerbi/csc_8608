## TP2 — Smoke test Diffusers (Stable Diffusion v1.5)

### Image générée (smoke test)

![alt text](../outputs/smoke.png)

## TP2 — Pipeline factorisé (text2img / img2img)

### Baseline text2img

![alt text](../outputs/baseline.png)

Configuration utilisée :
CONFIG: {'model_id': 'stable-diffusion-v1-5/stable-diffusion-v1-5', 'scheduler': 'EulerA', 'seed': 42, 'steps': 30, 'guidance': 7.5}

## TP2 — Expériences contrôlées (steps, guidance, scheduler)

Prompt (constant) : *ultra-realistic e-commerce product photo of a minimalist leather wallet on a clean white background, studio lighting, soft shadow, very sharp, 50mm lens*  
Seed (constant) : 42

### Résultats (6 runs)

| Run                                                      | Image                                  |
|----------------------------------------------------------|----------------------------------------|
| Run01 baseline (EulerA, steps=30, guidance=7.5)          | ![](../outputs/t2i_run01_baseline.png) |
| Run02 steps bas (EulerA, steps=15, guidance=7.5)         | ![](../outputs/t2i_run02_steps15.png)  |
| Run03 steps haut (EulerA, steps=50, guidance=7.5)        | ![](../outputs/t2i_run03_steps50.png)  |
| Run04 guidance bas (EulerA, steps=30, guidance=4.0)      | ![](../outputs/t2i_run04_guid4.png)    |
| Run05 guidance haut (EulerA, steps=30, guidance=12.0)    | ![](../outputs/t2i_run05_guid12.png)   |
| Run06 scheduler différent (DDIM, steps=30, guidance=7.5) | ![](../outputs/t2i_run06_ddim.png)     |

### Analyse qualitative (sans métrique)

- **Effet de `num_inference_steps`** :  
  - À **15 steps**, l’image converge moins : détails plus approximatifs, texture/contours parfois moins nets.  
  - À **50 steps**, on observe souvent plus de détails et une meilleure définition, mais le gain peut être marginal par rapport à 30 steps.

- **Effet de `guidance_scale`** :  
  - À **guidance=4.0**, l’image suit moins strictement le prompt : rendu parfois plus libre/moins fidèle, mais peut paraître plus naturel.  
  - À **guidance=12.0**, l’image colle plus au prompt mais peut devenir plus “forcée” : artefacts, rendu trop contrasté ou détails artificiels selon les cas.

- **Effet du scheduler** :  
  - Changer EulerA → DDIM modifie le style de convergence.  
  - À paramètres identiques (steps/guidance/seed), la composition globale reste proche, mais le rendu peut varier sensiblement.

## TP2 — Img2Img : effet du paramètre strength 

Image source (avant) :
![](../outputs/i2i_source.png)

Résultats :
- strength = 0.35  
![](../outputs/i2i_run07_strength035.png)

- strength = 0.60  
![](../outputs/i2i_run08_strength060.png)

- strength = 0.85  
![](../outputs/i2i_run09_strength085.png)

### Analyse qualitative

- **Ce qui est conservé (structure / identité)** :
  - À strength=0.35, la forme globale et le cadrage restent proches de l’image source ; le modèle retouche plus qu’il ne transforme.
  - À strength=0.60, l’identité produit est souvent encore reconnaissable, mais certains éléments peuvent changer.

- **Ce qui change (textures / fond / lumière / détails)** :
  - En augmentant strength, les textures deviennent plus générées, l’éclairage et les ombres peuvent être réinterprétés.
  - À strength=0.85, le fond et les détails peuvent être largement réinventés : le modèle s’éloigne fortement de l’image d’entrée.

- **Utilisabilité e-commerce** :
  - strength faible/moyen est généralement préférable pour rester fidèle au produit réel.
  - strength élevé (0.85) peut produire un rendu visuellement attractif mais trop loin du produit original : risque de modifier la forme, le matériau ou des caractéristiques importantes.

## TP2 — Mini-produit Streamlit (MVP)

L’application Streamlit permet :
- mode **Text2Img** (prompt → image),
- mode **Img2Img** (image + prompt → image),
- contrôle des paramètres : seed, steps, guidance, scheduler, et strength (img2img),
- affichage d’un bloc **Config** pour reproductibilité.

### Captures
- **Text2Img** : 

![alt text](../img/test2img.png)

- **Img2Img** : 

![alt text](../img/imgtoimg1.png)

![alt text](../img/imgtoimg.png)


## TP2 — Évaluation “light” (0–2) + total /10

### Grille (scores entiers 0–2)
- **Prompt adherence (0–2)** : 2 = correspond clairement au prompt, 0 = hors-sujet
- **Visual realism (0–2)** : 2 = photo crédible, 0 = rendu très artificiel
- **Artifacts (0–2)** : 2 = aucun artefact gênant, 0 = artefacts majeurs 
- **E-commerce usability (0–2)** : 2 = publiable après retouches mineures, 0 = inutilisable
- **Reproducibility (0–2)** : 2 = paramètres suffisants pour reproduire, 0 = infos manquantes

> Barème total : somme des 5 critères = **/10**.

---

### Image A — Text2Img baseline (référence)
**Fichier :** 
![](../outputs/baseline.png)

**Config (rappel) :** scheduler=EulerA, seed=42, steps=30, guidance=7.5, 512×512

| Critère | Score (0–2) |
|---|---|
| Prompt adherence | 2 |
| Visual realism | 2 |
| Artifacts | 1 |
| E-commerce usability | 2 |
| Reproducibility | 2 |
**Total : 9/10**

**Justification :**
- L’objet principal correspond au prompt et est bien isolé sur fond propre.
- Rendu global réaliste , mais petits détails perfectibles.
- Utilisable e-commerce après retouches légères, paramètres suffisants pour relancer.

---

### Image B — Text2Img “extrême” (paramètre extrême)
**Fichier :** 
![](../outputs/t2i_run05_guid12.png)

**Config (rappel) :** scheduler=EulerA, seed=42, steps=30, guidance=12.0, 512×512

| Critère | Score (0–2) |
|---|---|
| Prompt adherence | 2 |
| Visual realism | 1 |
| Artifacts | 1 |
| E-commerce usability | 1 |
| Reproducibility | 2 |
**Total : 7/10**

**Justification :**
- Le prompt est bien respecté mais le rendu peut devenir forcé ou au contraire moins fini.
- Augmentation des artefacts possibles.
- Utilisable seulement avec retouches plus importantes, mais reste reproductible grâce à la config.

---

### Image C — Img2Img strength élevé (obligatoire)
**Fichier :**  
**Avant :**  
![](../outputs/i2i_source.png)

**Après (strength=0.85) :**  
![](../outputs/i2i_run09_strength085.png)

**Config (rappel) :** scheduler=EulerA, seed=42, steps=30, guidance=7.5, strength=0.85, 512×512

| Critère | Score (0–2) |
|---|---|
| Prompt adherence | 2 |
| Visual realism | 1 |
| Artifacts | 1 |
| E-commerce usability | 0 |
| Reproducibility | 2 |
**Total : 6/10**

**Justification :**
- Le modèle suit bien le prompt mais s’éloigne fortement de l’image source : détails/texture/forme peuvent être réinventés.
- À strength élevé, la fidélité structurelle baisse : risque de changer”le produit.
- En e-commerce, cela devient risqué : utilisable plutôt pour inspiration, pas pour fiche produit.

---

## Réflexion (8–12 lignes)

Les paramètres influencent un compromis clair qualité vs latence/coût : augmenter num_inference_steps ou choisir certains schedulers améliore parfois la netteté et la stabilité, mais augmente linéairement le temps de génération et donc le coût GPU. À l’inverse, réduire steps accélère mais dégrade souvent les détails et augmente les artefacts.  
La reproductibilité dépend au minimum du couple, de la seed, des steps, du guidance, de la résolution. Elle peut casser si la version des librairies change, si le modèle est mis à jour côté hub, ou si on ne fixe pas correctement le générateur.  
En e-commerce, les risques principaux sont les hallucinations, des images trompeuses, et la conformité. Pour limiter ces risques, je privilégierais strength modéré, j’ajouterais des filtres, un contrôle humain, et je n’autoriserais la publication que sur des images validées ou retouchées, avec traçabilité complète des configs.
