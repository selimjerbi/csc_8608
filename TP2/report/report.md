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
