# Prisons

Gioco di editing di immagini: simulazione di personaggi in prigione in cui puoi creare scene animate modificando le immagini con **FLUX.2** in locale e usando [A2E Free, Personal AI Videos](https://a2e.ai/) per le animazioni video.

## Installazione

### Prerequisiti
1. Installa [Python 3.10](https://www.python.org/downloads/release/python-31011/) (versione 3.10.11, l'ultima con installer per Windows)
2. Installa [Cursor](https://cursor.com/download) (editor, opzionale)
3. Installa [Anaconda](https://www.anaconda.com/download)

### Setup
1. Clona il repository:
```bash
   git clone https://github.com/asprho-arkimete/prisos.git
   cd prisos
```
2. Crea l'ambiente virtuale:
```bash
   python -m venv vprisons
```
3. Attiva l'ambiente (Windows):
```bash
   vprisons\Scripts\activate
```
4. Installa PyTorch con CUDA:
```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
5. Installa le librerie:
```bash
   pip install -r requirements.txt
```
6. Avvia il gioco:
```bash
   python prisons.py
```
