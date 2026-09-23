 

from optparse import Values
import torch
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize
from deep_translator import GoogleTranslator
import os
from PIL import Image



nome_di="prison"
p=1
for d in range(1,9):
    os.makedirs(f"{nome_di} {d}",exist_ok=True)

def parse_risoluzione(ris_str):
    """
    Converte una stringa tipo '1:1 - 512x512' o '16:9 - 1500x843'
    in una tupla (width, height) di interi.
    """
    try:
        dims_part = ris_str.split(" - ")[-1]  # es. '512x512'
        w_str, h_str = dims_part.lower().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError, IndexError):
        # fallback di sicurezza
        return 1024, 1024

from deep_translator import GoogleTranslator, MyMemoryTranslator
def traduci_prompt(prompt, source='it', target='en'):
    # Tentativo 1: Google, intero blocco
    try:
        return GoogleTranslator(source=source, target=target).translate(prompt)
    except Exception as e:
        print(f"Google (blocco intero) fallito: {e}")

    # Tentativo 2: Google, frase per frase
    try:
        frasi = [f.strip() for f in prompt.split('.') if f.strip()]
        tradotte = [GoogleTranslator(source=source, target=target).translate(f) for f in frasi]
        return '. '.join(tradotte)
    except Exception as e:
        print(f"Google (per frase) fallito: {e}")

    # Tentativo 3: MyMemory (codici lingua-regione)
    try:
        return MyMemoryTranslator(source='it-IT', target='en-US').translate(prompt)
    except Exception as e:
        print(f"MyMemory fallito: {e}, uso testo originale")

    return prompt


def flux2(prompt, steps, path1, path2, path3, path4, risoluzione,path_Lora1, path_Lora2, outdir, outname):
    print("generazione image")
    device = "cuda"
    dtype = torch.bfloat16

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", dtype=dtype
    )

    if path_Lora1 and not path_Lora1.endswith(".safetensors"):
        path_Lora1 = f"{path_Lora1}.safetensors"

    if path_Lora2 and not path_Lora2.endswith(".safetensors"):
        path_Lora2 = f"{path_Lora2}.safetensors"

    if path_Lora1 and os.path.exists(path_Lora1):
        pipe.load_lora_weights(path_Lora1, adapter_name='Lora1')
        pipe.set_adapters('Lora1', adapter_weights=0.8)
    if path_Lora2 and os.path.exists(path_Lora2):
        pipe.load_lora_weights(path_Lora2, adapter_name='Lora2')
        pipe.set_adapters('Lora2', adapter_weights=0.8)

    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)
    quantize(pipe.text_encoder, weights=qfloat8)
    freeze(pipe.text_encoder)

    pipe.enable_model_cpu_offload()  # risparmia VRAM offloadando su CPU

    def f_resize(img, r):
        wr, hr = r
        w, h = img.size
        if w >= h:
            hr = (hr * h) // w
        else:
            wr = (wr * w) // h
        return img.resize((wr, hr), Image.Resampling.BICUBIC)

    paths = [path1, path2, path3, path4]
    n_valid = len([p for p in paths if p and os.path.exists(p)])
    target_res = 128 if n_valid > 2 else 512

    images = []
    for p in paths:
        if p and os.path.exists(p):
            img = Image.open(p).convert("RGB")
            images.append(f_resize(img, (target_res, target_res)))

    # Traduzione prompt da italiano a inglese
    translated = traduci_prompt(prompt, source='it', target='en')

    # Risolvi la risoluzione (width, height) dalla stringa della combobox
    width, height = parse_risoluzione(risoluzione)

    if len(images) == 0:
        image_arg = None
    elif len(images) == 1:
        image_arg = images[0]
    else:
        image_arg = images

    result = pipe(
        prompt=translated,
        image=image_arg,
        height=height,
        width=width,
        guidance_scale=1.0,
        num_inference_steps=int(steps),
        generator=torch.Generator(device=device).manual_seed(0)
    )
    image = result.images[0]

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    base, ext = os.path.splitext(outpath)
    k = 2
    while os.path.exists(outpath):
        outpath = f"{base}_{k}{ext}"
        k += 1

    image.save(outpath)
    return outpath


import tkinter as tk
from tkinter import Menu, ttk, filedialog
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES

window = TkinterDnD.Tk()   # necessario per abilitare il drag&drop
window.title("Prisons")
window.geometry("1920x1080")
window.state('zoomed')
window.resizable(False, False)

path_ref1 = None
path_ref2 = None

def f_genera_sfondi_oggetti():
    generate_object_window = tk.Toplevel(window)
    generate_object_window.title("GENERA OGGETTI E SFONDI")
    generate_object_window.geometry("820x520")
    generate_object_window.config(bg='gray')
    generate_object_window.resizable(False, False)
    generate_object_window.transient(window)

    # Opzionale: forza subito il focus sulla nuova finestra
    generate_object_window.lift()
    generate_object_window.focus_force()

    frame_c = tk.Frame(generate_object_window, bg='gray')
    frame_c.grid(row=0, column=0, sticky='nw', padx=10, pady=10)

    def carica_immagine(canvas, path, ref_key, label):
        """Carica e mostra l'immagine sul canvas, ridimensionata mantenendo l'aspect ratio."""
        img = Image.open(path)
        w, h = img.size
        rw, rh = 128, 128
        if w >= h:
            rh = (rw * h) // w
        else:
            rw = (rh * w) // h
        img = img.resize((rw, rh), Image.Resampling.BICUBIC)

        photo = ImageTk.PhotoImage(img)
        canvas.delete('all')
        canvas.create_image(64, 64, image=photo)
        canvas.image = photo  # mantieni il riferimento, altrimenti sparisce

        label.config(text=f"Ref{ref_key}: {os.path.basename(path)}")

        if ref_key == 1:
            global path_ref1
            path_ref1 = path
            print(f"Rif 1: {path_ref1}")
        else:
            global path_ref2
            path_ref2 = path
            print(f"Rif 2: {path_ref2}")
        

    def f_drop(event, canvas, ref_key, label):
        try:
            paths = window.tk.splitlist(event.data)
            if paths:
                carica_immagine(canvas, paths[0], ref_key, label)
        except Exception as e:
            print(f"Errore durante il drop: {e}")

    def f_click_browse(canvas, ref_key, label):
        path = filedialog.askopenfilename(
            filetypes=[('Immagini', '*.png *.jpg *.jpeg *.bmp *.gif')]
        )
        if path:
            carica_immagine(canvas, path, ref_key, label)

    # Canvas affiancati in alto a sinistra
    ref1 = tk.Canvas(frame_c, width=128, height=128, bg='light blue')
    ref1.create_text((64, 64), text='Trascina Foto\ndi Riferimento 1', font=("Arial", 9), width=120, justify='center')
    ref1.grid(row=0, column=0, sticky='nw')

    lab_canv = tk.Label(frame_c, text='Ref1: None', anchor='w', justify='left', wraplength=300)
    lab_canv.grid(row=0, column=1, sticky='w', padx=(5, 0))

    ref1.drop_target_register(DND_FILES)
    ref1.dnd_bind('<<Drop>>', lambda e: f_drop(e, ref1, 1, lab_canv))
    ref1.bind('<Double-Button-1>', lambda e: f_click_browse(ref1, 1, lab_canv))

    ref2 = tk.Canvas(frame_c, width=128, height=128, bg='light pink')
    ref2.create_text((64, 64), text='Trascina Foto\ndi Riferimento 2', font=("Arial", 9), width=120, justify='center')
    ref2.grid(row=0, column=2, sticky='nw', padx=(5, 0))

    lab_canv2 = tk.Label(frame_c, text='Ref2: None', anchor='w', justify='left', wraplength=300)
    lab_canv2.grid(row=0, column=3, sticky='w', padx=(5, 0))

    ref2.drop_target_register(DND_FILES)
    ref2.dnd_bind('<<Drop>>', lambda e: f_drop(e, ref2, 2, lab_canv2))
    ref2.bind('<Double-Button-1>', lambda e: f_click_browse(ref2, 2, lab_canv2))
    
    frame_text = tk.Frame(generate_object_window, bg='gray')
    frame_text.grid(row=1, column=0, sticky='nw', padx=10, pady=(0, 10))


    text_o = tk.Text(frame_text, width=70, height=13)
    text_o.grid(row=0, column=0, sticky='nw')

    frame_str = tk.Frame(generate_object_window, bg='gray')
    frame_str.grid(row=2, column=0, sticky='w')

    def s_steps(val):
        lbl_steps_o.config(text=f"Steps: {int(float(val))}")

    lbl_steps_o = tk.Label(frame_str, text="Steps:", fg='blue', bg='gray')
    lbl_steps_o.grid(row=1, column=0, sticky='w')

    steps_o = ttk.Scale(frame_str, from_=1, to=50, orient='horizontal', length=200, command=s_steps)
    steps_o.grid(row=2, column=0, sticky='nw', padx=10, pady=(0, 10))
    steps_o.set(8)

    def f_ris():
        risoluzioni = []
        for j in range(1, 5):
            dim = 512 * j
            risoluzioni.append(f"1:1 - {dim}x{dim}")
        for j in range(0, 11):
            w = 1000 + (100 * j)
            h = int(w * 9 / 16)
            risoluzioni.append(f"16:9 - {w}x{h}")
        return risoluzioni

    lbl_ris = tk.Label(frame_str, text="Risoluzione:", fg='dark green', bg='gray')
    lbl_ris.grid(row=1, column=1, sticky='w')

    risoluzione_cb = ttk.Combobox(frame_str, values=f_ris(), width=20)
    risoluzione_cb.grid(row=2, column=1, sticky='w', pady=(10, 0))
    risoluzione_cb.current(0)  # valore di default selezionato

    def f_load_Lora(event=None):
        models = ['nolora'] + [os.path.splitext(os.path.basename(m))[0] for m in os.listdir('lora')]
        lora1['values'] = models
        lora2['values'] = models

    lbl_Lora1 = tk.Label(frame_str, text="Lora 1", fg='dark green', bg='gray')
    lbl_Lora1.grid(row=1, column=2, sticky='w')
    lora1 = ttk.Combobox(frame_str, values=[], width=20)
    lora1.grid(row=2, column=2, sticky='w', pady=(10, 0))

    lbl_Lora2 = tk.Label(frame_str, text="Lora 2", fg='dark green', bg='gray')
    lbl_Lora2.grid(row=1, column=3, sticky='w')
    lora2 = ttk.Combobox(frame_str, values=[], width=20)
    lora2.grid(row=2, column=3, sticky='w', pady=(10, 0))

    f_load_Lora()          # popola le liste PRIMA di impostare l'indice
    lora1.current(0)
    lora2.current(0)

    lora1.bind('<Button-1>', f_load_Lora)
    lora2.bind('<Button-1>', f_load_Lora)

    def genera(outdir, outname):
        prompt = text_o.get("1.0", "end").strip()
        steps = int(float(steps_o.get()))
        risoluzione_val = risoluzione_cb.get()
        flux2(
            prompt=prompt,
            steps=steps,
            path1=path_ref1,
            path2=path_ref2,
            path3=None,
            path4=None,
            risoluzione=risoluzione_val,
            path_Lora1=f"lora//{lora1.get()}",
            path_Lora2=f"lora//{lora2.get()}",
            outdir=outdir,
            outname=outname
        )

    # genera images — command deve essere una funzione/lambda,
    # non il risultato di una chiamata immediata a flux2()
    Genera_oggetto = tk.Button(
        frame_str, bg='light blue', text='Genera Oggetto', width=20,
        command=lambda: genera('oggetti', 'oggetto.png')
    )
    Genera_oggetto.grid(row=3, column=0, sticky='nw', padx=10, pady=(0, 10))

    Genera_location = tk.Button(
        frame_str, bg='light green', text='Genera Location', width=20,
        command=lambda: genera('locations', 'location.png')
    )
    Genera_location.grid(row=3, column=1, sticky='w', pady=(10, 0))

import threading as T

import time
def mostra_su_canvas(outpath):
    """Mostra un'immagine sul canvas 'frame', ridimensionata e centrata."""
    if not outpath or not os.path.exists(outpath):
        frame.delete('all')
        return

    window.update_idletasks()  # forza il calcolo delle dimensioni reali dei widget
    cw, ch = frame.winfo_width(), frame.winfo_height()

    # Fallback di sicurezza: se il canvas non è ancora disegnato, usa le dimensioni configurate
    if cw <= 1 or ch <= 1:
        cw, ch = 1620, 980

    img = Image.open(outpath)
    w, h = img.size
    rw, rh = cw, ch
    if w >= h:
        rh = (cw * h) // w
    else:
        rw = (ch * w) // h

    # Ulteriore sicurezza: mai dimensioni a 0
    rw = max(rw, 1)
    rh = max(rh, 1)

    img = img.resize((rw, rh), Image.Resampling.BICUBIC)
    photo = ImageTk.PhotoImage(img)
    frame.delete('all')
    frame.create_image(cw // 2, ch // 2, image=photo)
    frame.image = photo
    f_modifica

path_ref1_C = None
path_ref2_C = None
path_ref3_C = None
path_ref4_C = None
lab_info_prison=None

canvas_ref1=None
canvas_ref2=None
canvas_ref3=None
canvas_ref4=None


genera_character_window_ref = None  # in cima al file, vicino a lab_info_prison=None

def f_genera_character():
    global path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C, lab_info_prison
    global canvas_ref1, canvas_ref2, canvas_ref3, canvas_ref4
    global genera_character_window_ref

    genera_character_window = tk.Toplevel(window)
    
    genera_character_window.title("Personaggi")
    genera_character_window.geometry("900x580")
    genera_character_window.config(bg='light gray')
    genera_character_window.resizable(False, False)

    # Rende la finestra "figlia" di window: resta sempre sopra alla sua finestra
    # padre (e viene minimizzata/ripristinata insieme ad essa), ma NON è
    # topmost rispetto ad altre applicazioni esterne (Cursor, Edge, Explorer...)
    genera_character_window.transient(window)

    # Opzionale: forza subito il focus sulla nuova finestra
    genera_character_window.lift()
    genera_character_window.focus_force()
    genera_character_window_ref = genera_character_window   # <-- salva il riferimento
    # ---------- RIGA 0: Canvas + Oggetti + Locations ----------
    frame_canvas = tk.Frame(genera_character_window, bg='light green')
    frame_canvas.grid(row=0, column=0, sticky='nw', padx=10, pady=10)

    def carica_immagine_C(canvas, path, ref_key):
        """Carica e mostra l'immagine sul canvas, ridimensionata mantenendo l'aspect ratio."""
        image = Image.open(path)
        rw, rh = 128, 128
        w, h = image.size
        if w >= h:
            rh = (rw * h) // w
        else:
            rw = (rh * w) // h

        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        photo = ImageTk.PhotoImage(image)

        canvas.delete('all')
        canvas.create_image(64, 64, image=photo)
        canvas.image = photo

        global path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C
        if ref_key == 1:
            path_ref1_C = path
        elif ref_key == 2:
            path_ref2_C = path
        elif ref_key == 3:
            path_ref3_C = path
        elif ref_key == 4:
            path_ref4_C = path

    def f_drop_C(event, canvas, ref_key):
        try:
            paths = window.tk.splitlist(event.data)
            if paths:
                carica_immagine_C(canvas, paths[0], ref_key)
        except Exception as e:
            print(f"Errore durante il drop: {e}")

    def f_click_browse_C(canvas, ref_key):
        path = filedialog.askopenfilename(
            filetypes=[('Immagini', '*.png *.jpg *.jpeg *.bmp *.gif')]
        )
        if path:
            carica_immagine_C(canvas, path, ref_key)

    canvas_ref1 = tk.Canvas(frame_canvas, width=128, height=128, bg='pink')
    canvas_ref1.create_text(64, 64, text="Trascina Foto\nriferimento 1", font=('Arial', 9))
    canvas_ref1.grid(row=0, column=0)
    canvas_ref1.drop_target_register(DND_FILES)
    canvas_ref1.dnd_bind('<<Drop>>', lambda e: f_drop_C(e, canvas_ref1, 1))
    canvas_ref1.bind('<Double-Button-1>', lambda e: f_click_browse_C(canvas_ref1, 1))

    canvas_ref2 = tk.Canvas(frame_canvas, width=128, height=128, bg='green')
    canvas_ref2.create_text(64, 64, text="Trascina Foto\nriferimento 2", font=('Arial', 9))
    canvas_ref2.grid(row=0, column=1)
    canvas_ref2.drop_target_register(DND_FILES)
    canvas_ref2.dnd_bind('<<Drop>>', lambda e: f_drop_C(e, canvas_ref2, 2))
    canvas_ref2.bind('<Double-Button-1>', lambda e: f_click_browse_C(canvas_ref2, 2))

    canvas_ref3 = tk.Canvas(frame_canvas, width=128, height=128, bg='light blue')
    canvas_ref3.create_text(64, 64, text="Trascina Foto\nriferimento 3", font=('Arial', 9))
    canvas_ref3.grid(row=1, column=0)
    canvas_ref3.drop_target_register(DND_FILES)
    canvas_ref3.dnd_bind('<<Drop>>', lambda e: f_drop_C(e, canvas_ref3, 3))
    canvas_ref3.bind('<Double-Button-1>', lambda e: f_click_browse_C(canvas_ref3, 3))

    canvas_ref4 = tk.Canvas(frame_canvas, width=128, height=128, bg='orange')
    canvas_ref4.create_text(64, 64, text="Trascina Foto\nriferimento 4", font=('Arial', 9))
    canvas_ref4.grid(row=1, column=1)
    canvas_ref4.drop_target_register(DND_FILES)
    canvas_ref4.dnd_bind('<<Drop>>', lambda e: f_drop_C(e, canvas_ref4, 4))
    canvas_ref4.bind('<Double-Button-1>', lambda e: f_click_browse_C(canvas_ref4, 4))

    # Oggetti — genitore corretto: genera_character_window, colonna 1
    frame_oggetti = tk.Frame(genera_character_window, bg='light gray')
    frame_oggetti.grid(row=0, column=1, sticky='nw', padx=(15, 0), pady=10)

    lab_oggetti = tk.Label(frame_oggetti, text='Oggetti', fg='blue', bg='light gray')
    lab_oggetti.grid(row=0, column=0, sticky='w')

    oggetti = ttk.Combobox(frame_oggetti, values=[], width=20)
    oggetti.grid(row=1, column=0, sticky='w')

    # Locations — genitore corretto: genera_character_window, colonna 2
    frame_locations = tk.Frame(genera_character_window, bg='light gray')
    frame_locations.grid(row=0, column=2, sticky='nw', padx=(15, 0), pady=10)

    lab_locations = tk.Label(frame_locations, text='Locations', fg='green', bg='light gray')
    lab_locations.grid(row=0, column=0, sticky='w')

    locations = ttk.Combobox(frame_locations, values=[], width=20)
    locations.grid(row=1, column=0, sticky='w')

    def load_oj_lo(dir):
        os.makedirs(dir, exist_ok=True)
        return [os.path.basename(f) for f in os.listdir(dir)]

    oggetti.bind('<Button-1>', lambda e: oggetti.config(values=load_oj_lo('oggetti')))
    locations.bind('<Button-1>', lambda e: locations.config(values=load_oj_lo('locations')))

    id_ogetto = 0
    id_location = 0

    def f_apply_selection(combobox, base_dir, tipo):
        nonlocal id_ogetto, id_location

        selezionato = combobox.get()
        if not selezionato:
            return

        path = os.path.join(base_dir, selezionato)
        if not os.path.exists(path):
            print(f"File non trovato: {path}")
            return

        canvases = {1: canvas_ref1, 2: canvas_ref2, 3: canvas_ref3, 4: canvas_ref4}
        paths = {1: path_ref1_C, 2: path_ref2_C, 3: path_ref3_C, 4: path_ref4_C}

        current_id = id_ogetto if tipo == 'oggetto' else id_location
        target_id = current_id if current_id != 0 else next((i for i in range(1, 5) if not paths[i]), 1)

        carica_immagine_C(canvases[target_id], path, target_id)

        if tipo == 'oggetto':
            id_ogetto = target_id
        else:
            id_location = target_id

    oggetti.bind('<<ComboboxSelected>>', lambda e: f_apply_selection(oggetti, 'oggetti', 'oggetto'))
    locations.bind('<<ComboboxSelected>>', lambda e: f_apply_selection(locations, 'locations', 'location'))

    # ---------- RIGA 1: Box di testo (a tutta larghezza) ----------
    frame_text = tk.Frame(genera_character_window, bg='gray')
    frame_text.grid(row=1, column=0, columnspan=3, sticky='w', padx=10, pady=(0, 10))

    text_o = tk.Text(frame_text, width=95, height=10)
    text_o.grid(row=0, column=0, sticky='nw')

    # ---------- RIGA 2: Steps / Risoluzione / Lora1 / Lora2 (a tutta larghezza) ----------
    frame_str = tk.Frame(genera_character_window, bg='gray')
    frame_str.grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=(0, 10))

    def s_steps(val):
        lbl_steps_o.config(text=f"Steps: {int(float(val))}")

    lbl_steps_o = tk.Label(frame_str, text="Steps: 8", fg='blue', bg='gray')
    lbl_steps_o.grid(row=0, column=0, sticky='w')

    steps_o = ttk.Scale(frame_str, from_=1, to=50, orient='horizontal', length=200, command=s_steps)
    steps_o.grid(row=1, column=0, sticky='nw', padx=10, pady=10)
    steps_o.set(8)

    def f_ris():
        risoluzioni = []
        for j in range(1, 5):
            dim = 512 * j
            risoluzioni.append(f"1:1 - {dim}x{dim}")
        for j in range(0, 11):
            w = 1000 + (100 * j)
            h = int(w * 9 / 16)
            w = round(w / 16) * 16
            h = round(h / 16) * 16
            risoluzioni.append(f"16:9 - {w}x{h}")
        return risoluzioni

    lbl_ris = tk.Label(frame_str, text="Risoluzione:", fg='dark green', bg='gray')
    lbl_ris.grid(row=0, column=1, sticky='w', padx=(15, 0))

    risoluzione_cb = ttk.Combobox(frame_str, values=f_ris(), width=20)
    risoluzione_cb.grid(row=1, column=1, sticky='w', padx=(15, 0))
    risoluzione_cb.set('16:9 - 1600x896')

    def f_load_Lora(event=None):
        os.makedirs('lora', exist_ok=True)
        models = ['nolora'] + [os.path.splitext(os.path.basename(m))[0] for m in os.listdir('lora')]
        lora1['values'] = models
        lora2['values'] = models

    lbl_Lora1 = tk.Label(frame_str, text="Lora 1", fg='dark green', bg='gray')
    lbl_Lora1.grid(row=0, column=2, sticky='w', padx=(15, 0))
    lora1 = ttk.Combobox(frame_str, values=[], width=20)
    lora1.grid(row=1, column=2, sticky='w', padx=(15, 0))

    lbl_Lora2 = tk.Label(frame_str, text="Lora 2", fg='dark green', bg='gray')
    lbl_Lora2.grid(row=0, column=3, sticky='w', padx=(15, 0))
    lora2 = ttk.Combobox(frame_str, values=[], width=20)
    lora2.grid(row=1, column=3, sticky='w', padx=(15, 0))

    f_load_Lora()
    lora1.current(0)
    lora2.current(0)

    lora1.bind('<Button-1>', f_load_Lora)
    lora2.bind('<Button-1>', f_load_Lora)




    def f_genera_c():
        print("Generazione personaggio")

        prompt = text_o.get("1.0", tk.END).strip()
        steps = int(steps_o.get())
        risoluzione_val = risoluzione_cb.get()
        path_Lora1 = f"lora//{lora1.get()}"
        path_Lora2 = f"lora//{lora2.get()}"

        p_corrente = p  # snapshot, evita race condition se l'utente cambia pagina durante la generazione
        ref1, ref2, ref3, ref4 = path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C

        def run():
            outpath = flux2(
                prompt=prompt,
                steps=steps,
                path1=ref1,
                path2=ref2,
                path3=ref3,
                path4=ref4,
                risoluzione=risoluzione_val,
                path_Lora1=path_Lora1,
                path_Lora2=path_Lora2,
                outdir=f'{nome_di} {p_corrente}',
                outname='prisoner.png'
            )
            # Aggiorna la GUI solo se l'utente è ancora sulla stessa prigione
            def aggiorna_gui():
                if p_corrente == p:
                    f_cronologia()  # ricarica la combobox e mostra la nuova immagine
            window.after(0, aggiorna_gui)

        T.Thread(target=run, daemon=True).start()


    genera_c = tk.Button(frame_str, text='Genera Character', bg='orange', command=f_genera_c)
    genera_c.grid(row=1, column=4, sticky='wn', padx=10, pady=5)

    # PARAMETRI PRIGIONE
    lab_info_prison=tk.Label(frame_str,text=f'Prigione {p} Vuota',fg='red')
    lab_info_prison.grid(row=2,column=1, sticky='wn', padx=10, pady=5)

os.makedirs("clips", exist_ok=True)

# ---------- Da mettere in cima a prisons.py (insieme agli altri import) ----------
import os
import re
import shutil
import cv2                      # pip install opencv-python
from PIL import Image, ImageTk
from tkinter import simpledialog

ESTENSIONI_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')
ESTENSIONI_IMMAGINE = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
ESTENSIONI_SUPPORTATE = ESTENSIONI_IMMAGINE + ESTENSIONI_VIDEO

RE_CLIP = re.compile(r'^clips(\d+)\.[^.]+$', re.IGNORECASE)   # clips3.mp4, clips12.png, ...


def chiave_ordinamento(nome):
    """Ordine numerico (clips2 prima di clips10); i file con altri nomi vanno in fondo."""
    m = RE_CLIP.match(nome)
    return (0, int(m.group(1)), '') if m else (1, 0, nome.lower())


def elenco_clips(cartella='clips'):
    """Restituisce {indice: [nomi file]} per i file chiamati clips<N>.<estensione>."""
    trovati = {}
    for nome in os.listdir(cartella):
        m = RE_CLIP.match(nome)
        if m and nome.lower().endswith(ESTENSIONI_SUPPORTATE):
            trovati.setdefault(int(m.group(1)), []).append(nome)
    return trovati


def libera_indice(indice, cartella='clips'):
    """Se clips<indice> esiste, la sposta a indice+1; se anche quello esiste, sposta
    pure lui, e così via finché non trova un indice libero (poi si ferma)."""
    trovati = elenco_clips(cartella)
    fine = indice
    while fine in trovati:          # primo indice libero a partire da 'indice'
        fine += 1
    # rinomino dal più alto al più basso, così non sovrascrivo mai nulla
    for k in range(fine - 1, indice - 1, -1):
        for nome in trovati[k]:
            estensione = os.path.splitext(nome)[1]
            os.rename(os.path.join(cartella, nome),
                      os.path.join(cartella, f'clips{k + 1}{estensione}'))


# ---------- Funzione timeline() con scale per i fotogrammi ----------
def timeline():
    Time_line = tk.Toplevel(window)
    Time_line.title("Timeline")
    Time_line.geometry("1200x580")
    Time_line.config(bg='light green')
    Time_line.resizable(False, False)
    Time_line.transient(window)
    Time_line.lift()
    Time_line.focus_force()

    # ---------- Contenitore scrollabile orizzontale ----------
    # la colonna 0 della finestra deve poter crescere, altrimenti il frame
    # resta grande quanto il suo contenuto invece di riempire la finestra
    Time_line.grid_columnconfigure(0, weight=1)

    frame_esterno = tk.Frame(Time_line, bg='light green')
    frame_esterno.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

    # width = 1200 (finestra) - 2*10 (padx) = 1180: quasi tutta la larghezza di Time_line
    # altezza 400: sotto ogni canvas ci sono la scala e il bottone di salvataggio
    scroll_canvas = tk.Canvas(frame_esterno, width=1180, height=400, bg='gray', highlightthickness=0)
    scrollbar_h = tk.Scrollbar(frame_esterno, orient='horizontal', command=scroll_canvas.xview)
    scroll_canvas.configure(xscrollcommand=scrollbar_h.set)

    scroll_canvas.pack(side='top', fill='both', expand=True)
    scrollbar_h.pack(side='top', fill='x')

    # Il "raccoglitore": il frame che ospita gli slot (canvas + scala) in fila
    pannel_raccoglitore = tk.Frame(scroll_canvas, bg='gray')
    scroll_canvas.create_window((0, 0), window=pannel_raccoglitore, anchor='nw')

    def _on_frame_configure(event=None):
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))

    pannel_raccoglitore.bind('<Configure>', _on_frame_configure)

    # Scroll orizzontale con Shift + rotellina
    def _on_mousewheel(event):
        scroll_canvas.xview_scroll(int(-1 * (event.delta / 120)), 'units')

    scroll_canvas.bind_all('<Shift-MouseWheel>', _on_mousewheel)

    # Lista dei riferimenti alle canvas create (utile per accedervi dopo)
    lista_canvas_clip = []

    # bind_all è globale: lo rimuovo alla chiusura, e rilascio i video aperti
    def _on_close(event=None):
        if event is None or event.widget is Time_line:
            try:
                Time_line.unbind_all('<Shift-MouseWheel>')
            except tk.TclError:
                pass
            for c in lista_canvas_clip:
                cap = getattr(c, 'cap', None)
                if cap is not None:
                    cap.release()
                    c.cap = None

    Time_line.bind('<Destroy>', _on_close)

    # ---------- Visualizzazione ----------
    def mostra_pil(canvas, img):
        """Ridimensiona (max 256x256, proporzioni mantenute) e mostra nella canvas."""
        w, h = img.size
        rw, rh = 256, 256
        if w >= h:
            rh = max(1, (rw * h) // w)
        else:
            rw = max(1, (rh * w) // h)
        img = img.resize((rw, rh), Image.Resampling.BICUBIC)
        photo = ImageTk.PhotoImage(img)
        canvas.delete('all')
        canvas.create_image(128, 128, image=photo)
        canvas.image = photo   # mantieni il riferimento

    def vai_al_frame(canvas, valore):
        """Callback della scala: mostra il fotogramma richiesto del video."""
        cap = getattr(canvas, 'cap', None)
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(valore)))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mostra_pil(canvas, Image.fromarray(frame))

    def salva_frame(canvas):
        """Salva in frames/<nome file>_<numero frame>.png il fotogramma scelto con la scala."""
        path = getattr(canvas, 'path', None)
        if path is None:
            print("Nessuna clip caricata in questa canvas")
            return

        n = int(canvas.scala.get())   # numero del fotogramma selezionato

        try:
            cap = getattr(canvas, 'cap', None)
            if cap is not None:
                # video: rileggo il fotogramma a piena risoluzione (non quello ridimensionato)
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"impossibile leggere il fotogramma {n}")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # immagine singola: viene salvata come fotogramma 0
                img = Image.open(path).convert('RGB')

            os.makedirs('frames', exist_ok=True)
            nome = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join('frames', f"{nome}_{n}.png")
            img.save(out)
            print(f"Fotogramma salvato in {out}")

            # piccolo feedback visivo sul bottone
            btn = canvas.btn_salva
            btn.config(text='Salvato ✓')
            btn.after(1200, lambda: btn.config(text='Salva fotogramma'))
        except Exception as e:
            print(f"Errore nel salvataggio del fotogramma: {e}")

    def imposta_scala(canvas, n_frames):
        """Configura la scala della canvas: attiva per i video, disattivata per le immagini."""
        scala = canvas.scala
        if n_frames > 1:
            scala.config(from_=0, to=n_frames - 1, state='normal')
        else:
            scala.config(from_=0, to=0, state='disabled')
        scala.set(0)

    # ---------- Caricamento ----------
    def carica_immagine_clip(canvas, path):
        """Carica un'immagine, oppure un video (primo fotogramma + scala attiva)."""
        # rilascia un eventuale video caricato in precedenza in questa canvas
        vecchio = getattr(canvas, 'cap', None)
        if vecchio is not None:
            vecchio.release()
            canvas.cap = None

        try:
            if path.lower().endswith(ESTENSIONI_VIDEO):
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise ValueError("impossibile aprire il video")
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    raise ValueError("impossibile leggere il primo fotogramma")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                canvas.cap = cap
            else:
                img = Image.open(path).convert('RGB')
                n_frames = 1
        except Exception as e:
            print(f"Impossibile caricare {path}: {e}")
            return False

        mostra_pil(canvas, img)
        canvas.path = path
        imposta_scala(canvas, n_frames)
        canvas.btn_salva.config(state='normal')
        return True

    def f_drop_clip(event, canvas):
        try:
            paths = window.tk.splitlist(event.data)
            if paths:
                carica_immagine_clip(canvas, paths[0])
        except Exception as e:
            print(f"Errore durante il drop: {e}")

    def f_click_browse_clip(canvas):
        path = filedialog.askopenfilename(
            filetypes=[
                ('Immagini e video',
                 '*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.webm *.wmv'),
                ('Immagini', '*.png *.jpg *.jpeg *.bmp *.gif'),
                ('Video', '*.mp4 *.avi *.mov *.mkv *.webm *.wmv'),
            ]
        )
        if path:
            carica_immagine_clip(canvas, path)

    def crea_canvas_clip():
        """Crea uno slot (canvas + scala) nel raccoglitore, pronto per drag&drop."""
        idx = len(lista_canvas_clip)

        # ogni slot è un frame con la canvas sopra e la scala sotto
        slot = tk.Frame(pannel_raccoglitore, bg='gray')
        slot.grid(row=0, column=idx, padx=5, pady=5)

        canvas_clip = tk.Canvas(slot, width=256, height=256, bg='red')
        canvas_clip.create_text(
            128, 128,
            text="Trascina qui\nun'immagine o un video\n(doppio click per sfogliare)",
            font=('Arial', 10), justify='center'
        )
        canvas_clip.grid(row=0, column=0)

        scala = tk.Scale(
            slot, from_=0, to=0, orient='horizontal', length=256,
            state='disabled', bg='gray',
            command=lambda v, c=canvas_clip: vai_al_frame(c, v)
        )
        scala.grid(row=1, column=0, pady=(4, 0))

        btn_salva = tk.Button(
            slot, text='Salva fotogramma', state='disabled',
            command=lambda c=canvas_clip: salva_frame(c)
        )
        btn_salva.grid(row=2, column=0, pady=(4, 0))

        canvas_clip.scala = scala
        canvas_clip.btn_salva = btn_salva
        canvas_clip.cap = None

        canvas_clip.drop_target_register(DND_FILES)
        canvas_clip.dnd_bind('<<Drop>>', lambda e, c=canvas_clip: f_drop_clip(e, c))
        canvas_clip.bind('<Double-Button-1>', lambda e, c=canvas_clip: f_click_browse_clip(c))

        lista_canvas_clip.append(canvas_clip)
        return canvas_clip

    def rilascia_video():
        """Chiude tutti i video aperti (su Windows un file aperto non si può rinominare)."""
        for c in lista_canvas_clip:
            cap = getattr(c, 'cap', None)
            if cap is not None:
                cap.release()
                c.cap = None

    def ricarica_clips():
        """Svuota la timeline e ricarica tutte le clip dalla cartella clips, in ordine di indice."""
        rilascia_video()
        for c in lista_canvas_clip:
            c.master.destroy()      # distrugge lo slot (canvas + scala + bottone)
        lista_canvas_clip.clear()

        os.makedirs('clips', exist_ok=True)
        for nome_file in sorted(os.listdir('clips'), key=chiave_ordinamento):
            path_file = os.path.join('clips', nome_file)
            if os.path.isfile(path_file) and path_file.lower().endswith(ESTENSIONI_SUPPORTATE):
                c = crea_canvas_clip()
                carica_immagine_clip(c, path_file)

    def aggiungi_nuova_clip():
        """Copia un file nella cartella clips come clips<indice>.<estensione>,
        spostando in avanti le clip già presenti con quell'indice."""
        sorgente = filedialog.askopenfilename(
            parent=Time_line,
            title='Scegli la clip da aggiungere',
            filetypes=[
                ('Immagini e video',
                 '*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.webm *.wmv'),
                ('Tutti i file', '*.*'),
            ]
        )
        if not sorgente:
            return

        estensione = os.path.splitext(sorgente)[1].lower()
        if estensione not in ESTENSIONI_SUPPORTATE:
            print(f"Estensione non supportata: {estensione}")
            return
        if os.path.dirname(os.path.abspath(sorgente)) == os.path.abspath('clips'):
            print("Il file è già nella cartella clips: scegline uno da un'altra cartella")
            return

        os.makedirs('clips', exist_ok=True)
        proposto = max(elenco_clips().keys(), default=0) + 1
        indice = simpledialog.askinteger(
            'Indice clip', 'Indice della clip (es. 3 per "clips3"):',
            parent=Time_line, minvalue=0, initialvalue=proposto
        )
        if indice is None:
            return

        try:
            rilascia_video()                 # altrimenti Windows blocca il rename
            libera_indice(indice)
            destinazione = os.path.join('clips', f'clips{indice}{estensione}')
            shutil.copy2(sorgente, destinazione)
            print(f"Clip salvata in {destinazione}")
        except Exception as e:
            print(f"Errore nell'aggiunta della clip: {e}")
        finally:
            ricarica_clips()                 # ricostruisce la timeline con i nuovi nomi

    # Carica le clip già salvate su disco all'apertura della finestra
    ricarica_clips()

    # ---------- Bottone per aggiungere una nuova canvas ----------
    frame_bottoni = tk.Frame(Time_line, bg='light green')
    frame_bottoni.grid(row=1, column=0, sticky='w', padx=10, pady=(0, 10))

    aggiungi_nuova_canvas = tk.Button(
        frame_bottoni,
        text='Aggiungi nuova canvas',
        command=crea_canvas_clip
    )
    aggiungi_nuova_canvas.grid(row=0, column=0)

    btn_nuova_clip = tk.Button(
        frame_bottoni,
        text='Aggiungi nuova clip',
        command=aggiungi_nuova_clip
    )
    btn_nuova_clip.grid(row=0, column=1, padx=(10, 0))


# Apertura diretta delle finestre (con thread, Tkinter gestisce già l'asincronia)
T.Thread(target=f_genera_sfondi_oggetti(),daemon=True)

T.Thread(target=f_genera_character(),daemon=True)

T.Thread(target=timeline(),daemon=True).start()


menu_bar = Menu(window)

file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Save")
file_menu.add_command(label="Save As")
menu_bar.add_cascade(label="File", menu=file_menu)

tools_menu = Menu(menu_bar, tearoff=0)
tools_menu.add_command(label="Lora")
tools_menu.add_command(label="Character Generate",command= f_genera_character)
tools_menu.add_command(label="Object,location Generate",command=f_genera_sfondi_oggetti)
menu_bar.add_cascade(label="Tools", menu=tools_menu)

window.config(menu=menu_bar)

p = 1        # pagina corrente
MIN_P = 1
MAX_P = 8


def f_cronologia():
    """Popola la combobox con le immagini della prigione corrente
    e mostra la prima immagine (o quella già selezionata, se coerente)."""
    global p
    cartella = f"{nome_di} {p}"
    os.makedirs(cartella, exist_ok=True)

    models = [c for c in os.listdir(cartella) if os.path.isfile(os.path.join(cartella, c))]
    combo_cronologia['values'] = models

    if models:
        combo_cronologia.current(0)  # seleziona sempre la prima immagine disponibile
        mostra_su_canvas(os.path.join(cartella, models[0]))
    else:
        combo_cronologia.set('')
        mostra_su_canvas(None)  # nessuna immagine, pulisce il canvas


def f_seleziona_cronologia(event=None):
    """Chiamata quando l'utente sceglie manualmente un'immagine dalla combobox."""
    cartella = f"{nome_di} {p}"
    selezionato = combo_cronologia.get()
    if selezionato:
        mostra_su_canvas(os.path.join(cartella, selezionato))


def aggiorna_bottoni():
    if p > MIN_P:
        prison_precedente.config(text=f"Prison {p - 1}", state='normal')
    else:
        prison_precedente.config(text="Prison -", state='disabled')

    if p < MAX_P:
        prison_successiva.config(text=f"Prison {p + 1}", state='normal')
    else:
        prison_successiva.config(text="Prison -", state='disabled')

def upgrate_parametri():
    global lab_info_prison
    try:
        if not lab_info_prison.winfo_exists():
            return
    except tk.TclError:
        return

    if len(os.listdir(f"{nome_di} {p}")) > 0:
        lab_info_prison.config(text=f'Prigione {p} Piena', fg='green')
    else:
        lab_info_prison.config(text=f'Prigione {p} Vuota', fg='red')
    lab_info_prison.update_idletasks()

upgrate_parametri()

def f_p_prec():
    global p
    if p > MIN_P:
        p -= 1
        aggiorna_bottoni()
        f_cronologia()
        upgrate_parametri()


def f_p_succes():
    global p
    if p < MAX_P:
        p += 1
        aggiorna_bottoni()
        f_cronologia()
        upgrate_parametri()


prison_precedente = tk.Button(window, text="", command=f_p_prec)
prison_precedente.grid(row=1, column=0, sticky='wn')

frame = tk.Canvas(window, width=1620, height=980, bg='red')
frame.grid(row=1, column=0, padx=60, pady=4)

frame_st2=tk.Frame(window)
frame_st2.grid(row=2,column=0,sticky='wn',padx=10,pady=2)

combo_cronologia = ttk.Combobox(frame_st2, values=[])
combo_cronologia.grid(row=0, column=0, padx=20, pady=4)
combo_cronologia.bind('<<ComboboxSelected>>', f_seleziona_cronologia)

#agiunggi checkbok modifica image prigione
id_use = 0
modifica_image_prigione_var = tk.BooleanVar(value=False)

def carica_immagine_C(canvas, path, ref_key):
    """Carica e mostra l'immagine sul canvas, ridimensionata mantenendo l'aspect ratio."""
    image = Image.open(path)
    rw, rh = 128, 128
    w, h = image.size
    if w >= h:
        rh = (rw * h) // w
    else:
        rw = (rh * w) // h

    image = image.resize((rw, rh), Image.Resampling.BICUBIC)
    photo = ImageTk.PhotoImage(image)

    canvas.delete('all')
    canvas.create_image(64, 64, image=photo)
    canvas.image = photo

    global path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C
    if ref_key == 1:
        path_ref1_C = path
    elif ref_key == 2:
        path_ref2_C = path
    elif ref_key == 3:
        path_ref3_C = path
    elif ref_key == 4:
        path_ref4_C = path


def _reset_canvas(ref_key):
    """Svuota il canvas indicato e azzera il path corrispondente."""
    global path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C

    canvas_map = {
        1: canvas_ref1,
        2: canvas_ref2,
        3: canvas_ref3,
        4: canvas_ref4,
    }
    canvas = canvas_map.get(ref_key)
    if canvas is None:
        return

    canvas.delete('all')
    canvas.create_text(64, 64, text=f"Trascina Foto\nriferimento {ref_key}", font=('Arial', 9))
    canvas.image = None

    if ref_key == 1:
        path_ref1_C = None
    elif ref_key == 2:
        path_ref2_C = None
    elif ref_key == 3:
        path_ref3_C = None
    elif ref_key == 4:
        path_ref4_C = None


def f_modifica():
    """Se la checkbox è attiva, cerca il primo canvas con path mancante
    o inesistente e vi applica l'immagine della prigione corrispondente
    alla cronologia selezionata. Se viene disattivata, rimuove l'immagine
    di prigione dal canvas occupato."""
    global path_ref1_C, path_ref2_C, path_ref3_C, path_ref4_C
    global canvas_ref1, canvas_ref2, canvas_ref3, canvas_ref4
    global id_use

    if not modifica_image_prigione_var.get():
        if id_use:
            _reset_canvas(id_use)
            id_use = 0
        return

    # Costruisci il path dell'immagine della prigione
    # (combo_cronologia.get() è già il nome file completo, con estensione)
    selezionato = combo_cronologia.get()
    if not selezionato:
        print("Nessuna immagine di cronologia selezionata")
        modifica_image_prigione_var.set(False)
        return

    img_path = os.path.join(f"{nome_di} {p}", selezionato)

    if not os.path.exists(img_path):
        print(f"Immagine di prigione non trovata: {img_path}")
        modifica_image_prigione_var.set(False)
        return

    canvases = [
        (canvas_ref1, path_ref1_C, 1),
        (canvas_ref2, path_ref2_C, 2),
        (canvas_ref3, path_ref3_C, 3),
        (canvas_ref4, path_ref4_C, 4),
    ]

    for canvas, path, ref_key in canvases:
        # path mancante (None) OPPURE file non più esistente sul disco
        if path is None or not os.path.exists(path):
            carica_immagine_C(canvas, img_path, ref_key)
            id_use = ref_key
            break  # occupa solo il primo slot libero trovato
    else:
        print("Tutti i canvas hanno già un'immagine valida, nessuno slot libero")
        modifica_image_prigione_var.set(False)


modifica_image_prigione = ttk.Checkbutton(
    frame_st2,
    text=f'usa prigione {p} come riferimento',
    variable=modifica_image_prigione_var,
    command=f_modifica
)
modifica_image_prigione.grid(row=0, column=1)

prison_successiva = tk.Button(window, text="", command=f_p_succes)
prison_successiva.grid(row=1, column=2, sticky='wn')

aggiorna_bottoni()
f_cronologia()  

# ORA chiamata dopo che combo_cronologia esiste: mostra la prima immagine all'avvio
after_ids = []
def chiudi_finestra():
    for aid in after_ids:
        try:
            genera_character_window_ref.after_cancel(aid)
        except Exception:
            pass
    genera_character_window_ref.destroy()
genera_character_window_ref.protocol("WM_DELETE_WINDOW", chiudi_finestra)



# come ultimissima riga del file
window.mainloop()
