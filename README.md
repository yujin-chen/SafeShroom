# SafeShroom: Toxicity-Centric Mushroom Identification 
**SafeShroom** is a deep learning computer vision project developed for the **ICS 483 Final Project**. It aims to solve the critical safety issue in automated mushroom identification: "silent failures" where toxic species are misclassified as edible.

Unlike traditional apps that classify species first and look up toxicity in a database, SafeShroom uses a **Multi-Task Learning (MTL)** framework to simultaneously predict species taxonomy and toxicity. By forcing the CNN to learn fine-grained physical features, we create a "safety guardrail" that improves toxicity detection and reduces dangerous false alarms.



## Motivation
Mushroom foraging is gaining popularity, but misidentification can be lethal. Many toxic species, such as the *Death Cap*, utilize Batesian mimicry to resemble edible varieties. 

Traditional "lookup" classifiers fail if they misclassify the species name. SafeShroom addresses this by treating **Toxicity Detection** as a primary visual task. It learns to recognize morphological danger signals (e.g., volvas, warts) directly from the image, providing a safety net even when the exact species is unknown.

---

## Key Features
* **Multi-Task Learning (MTL):** Simultaneous training on Species Classification (182 classes) and Toxicity Classification (Binary).
* **Safety Guardrail:** Optimized specifically to minimize False Negatives (missed toxic mushrooms) and False Positives (edible mushrooms flagged as toxic).
* **Object-Centric Pipeline:** Integrated YOLOv8-X preprocessing to crop mushrooms from background clutter, focusing attention on morphology.
* **Explainable AI:** Grad-CAM integration to visualize which parts of the mushroom (cap, stem, gills) the model uses to make safety decisions.

---

## Architecture
The system uses a shared **EfficientNet-B3** backbone to extract visual features. These embeddings are split into two parallel heads:
1.  **Species Head:** Classifies the specific mushroom name.
2.  **Toxicity Head:** A binary classifier optimized with weighted loss to detect danger.


---

## Dataset
Utilize the **Danish Fungi 2020 (DF20) / FungiCLEF 2024** dataset.
* **Size:** ~36,000 images ("Mini" subset)
* **Classes:** 182 unique species
* **Preprocessing:** Auto-cropped using fine-tuned YOLOv8; resized to 300x300.

---

