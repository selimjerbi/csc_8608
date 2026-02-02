# TP1 — Setup (repo / GPU / UI)

## Dépôt
- Repo : https://github.com/selimjerbi/csc_8608.git

## Environnement d’exécution
- Exécution : nœud GPU via SLURM (cluster TSP) 

## Arborescence

![alt text](../img/arbo.png)

## Conda + CUDA

![alt text](../img/env.png)

## Dépendances

![alt text](../img/dep.png)

## UI Streamlit via SSH
- Port choisi : `8511`
- UI accessible via SSH tunnel : oui

![alt text](../img/curl_stramlit.png)

## Mini-dataset d’images

- Nombre final d’images : **8**
- Sources : images récupérées via recherche web

### Images représentatives
1. `im1.jpeg` — Objet unique bien visible sur fond simple (cas simple, segmentation facile).
2. `im2.jpeg` — Objet principal complexe mais isolé (PC), bon contraste global.
3. `im4.jpeg` — Scène de rue avec plusieurs objets et arrière-plan chargé (cas complexe).
4. `im6.jpeg` — Cuisine avec de nombreux éléments et plans visuels (cas chargé).
5. `im7.jpeg` — Grillage fin et répétitif, contours difficiles à segmenter (cas difficile).

### Exemples  
- Cas simple :

![alt text](../img/streamlit_im1.png)

- Cas difficile :

![alt text](../img/stramlit_im8.png)

## SAM — Chargement GPU et inférence bbox → masque

- **Modèle SAM utilisé** : `vit_b`
- **Checkpoint** : `sam_vit_b_01ec64.pth`  

### Test rapide (preuve d’exécution)
Sortie console :
- device : `cuda`
- image : `im6.jpeg`
- image shape : `(189, 267, 3)` (RGB, uint8)
- mask shape : `(189, 267)`
- score : `0.893`
- mask_sum : `17515`

![alt text](../img/sortie_sam.png)

### Premiers constats
L’inférence fonctionne correctement : le modèle se charge sur GPU et produit un masque binaire de la même résolution que l’image d’entrée.  
Avec le modèle `vit_b`, le temps d’exécution est rapide et compatible avec une utilisation interactive via l’interface Streamlit.  
La qualité du masque dépend fortement de la bounding box fournie : une box approximative peut inclure des zones non pertinentes.  
Le mode `multimask` est utile pour les cas ambigus, mais un choix automatique du meilleur masque reste nécessaire.
