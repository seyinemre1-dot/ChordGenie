import os
import sys
import subprocess
import re
import difflib

# --- OTO-KURULUM VE KÜTÜPHANE KONTROLÜ ---
try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

import streamlit as st
import librosa
import numpy as np
from faster_whisper import WhisperModel
import torch
import scipy.signal
import yt_dlp

# --- 1. SABİTLER VE SİSTEM ---
NOTALAR_STANDART = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
MAP_12 = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11}
INDEX_TO_KEY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Halüsinasyon Zırhı
KARA_LISTE = ["müzik", "music", "bgm", "instrumental", "alkış", "gülme", "altyazı", "mk", "mustafa", "köse", "m", "k", "dimatorzok", "субтитры", "сделал"]

# Akor Şablonları (Template Matching) - Kök nota öncelikli!
MAJ_TEMPLATE = np.array([1.5, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
MIN_TEMPLATE = np.array([1.5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
CHORD_TEMPLATES = []
CHORD_LABELS = []

for i in range(12):
    CHORD_TEMPLATES.append(np.roll(MAJ_TEMPLATE, i))
    CHORD_LABELS.append(INDEX_TO_KEY[i])
    CHORD_TEMPLATES.append(np.roll(MIN_TEMPLATE, i))
    CHORD_LABELS.append(INDEX_TO_KEY[i] + 'm')

CHORD_TEMPLATES = np.array(CHORD_TEMPLATES)
CHORD_TEMPLATES_NORM = CHORD_TEMPLATES / np.linalg.norm(CHORD_TEMPLATES, axis=1, keepdims=True)

# --- 2. YARDIMCI FONKSİYONLAR VE SVG ÇİZİM MOTORU ---
@st.cache_resource
def get_fast_model():
    return WhisperModel("turbo", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8_float16")

def transpose_chord(chord, orig, target):
    if not chord: return ""
    steps = (MAP_12[target] - MAP_12[orig]) % 12
    is_minor = chord.endswith('m')
    root = chord[:-1] if is_minor else chord
    if root in MAP_12:
        new_idx = (MAP_12[root] + steps) % 12
        return f"{INDEX_TO_KEY[new_idx]}{'m' if is_minor else ''}"
    return chord

def get_chord_svg(chord_name):
    chord_map = {
        'C': 'x32010', 'C#': 'x46664', 'Db': 'x46664', 'D': 'xx0232', 'D#': 'xx1343', 'Eb': 'xx1343',
        'E': '022100', 'F': '133211', 'F#': '244322', 'Gb': '244322', 'G': '320003', 'G#': '466544', 'Ab': '466544',
        'A': 'x02220', 'A#': 'x13331', 'Bb': 'x13331', 'B': 'x24442',
        'Cm': 'x35543', 'C#m': 'x46654', 'Dbm': 'x46654', 'Dm': 'xx0231', 'D#m': 'xx1342', 'Ebm': 'xx1342',
        'Em': '022000', 'Fm': '133111', 'F#m': '244222', 'Gbm': '244222', 'Gm': '355333', 'G#m': '466444', 'Abm': '466444',
        'Am': 'x02210', 'A#m': 'x13321', 'Bbm': 'x13321', 'Bm': 'x24432'
    }
    frets = chord_map.get(chord_name)
    if not frets: return ""

    fret_nums = [int(x) for x in frets if x != 'x']
    min_fret = min(fret_nums) if fret_nums and min(fret_nums) > 0 else 1
    start_fret = min_fret if min_fret > 2 else 1

    svg = '<svg width="50" height="65" viewBox="0 0 50 65" xmlns="http://www.w3.org/2000/svg">'
    for i in range(6): 
        x = 5 + i * 8
        svg += f'<line x1="{x}" y1="15" x2="{x}" y2="60" stroke="#555" stroke-width="1"/>'
    for i in range(5): 
        y = 15 + i * 11.25
        w = 2 if (i == 0 and start_fret == 1) else 1
        svg += f'<line x1="5" y1="{y}" x2="45" y2="{y}" stroke="#555" stroke-width="{w}"/>'

    if start_fret > 1: 
        svg += f'<text x="0" y="25" font-family="sans-serif" font-size="8" fill="#555">{start_fret}</text>'

    for i, fret in enumerate(frets): 
        x = 5 + i * 8
        if fret == 'x':
            svg += f'<text x="{x-3}" y="10" font-family="sans-serif" font-size="10" fill="#d9534f">x</text>'
        elif fret == '0':
            svg += f'<circle cx="{x}" cy="10" r="2" fill="none" stroke="#555" stroke-width="1"/>'
        else:
            f = int(fret)
            rel_fret = f - start_fret
            y = 15 + rel_fret * 11.25 + 5.6
            svg += f'<circle cx="{x}" cy="{y}" r="3.5" fill="#d9534f"/>'
    svg += '</svg>'
    return svg

def download_audio_from_url(url):
    os.environ["YTDLP_JS_RUNTIME"] = "node" 
    if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': 'temp_audio', 
        'quiet': True, 'noplaylist': True, 'nocheckcertificate': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info.get('title', 'Şarkı')

def create_pdf(analiz_verisi, orig_key, target_key):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=12) 
    
    pdf.cell(200, 10, txt="Studio Dashboard Pro - Repertuvar", ln=True, align='C')
    pdf.ln(10)

    for item in analiz_verisi:
        words = item[1] if len(item) > 1 else []
        count = item[2] if len(item) > 2 else 1
        
        akor_satiri = ""
        soz_satiri = ""
        
        for w in words:
            t_chord = transpose_chord(w.get("a", ""), orig_key, target_key)
            kelime = w.get("k", "")
            
            def tr_fix(text):
                tr_map = str.maketrans("ĞĞğğŞŞşşİİııÖÖööÜÜüüÇÇçç", "GGggSSssIIiiOOooUUuuCCcc")
                return text.translate(tr_map)
            
            t_chord = tr_fix(t_chord)
            kelime = tr_fix(kelime)
            
            max_len = max(len(t_chord), len(kelime)) + 2
            akor_satiri += t_chord.ljust(max_len)
            soz_satiri += kelime.ljust(max_len)
            
        pdf.set_text_color(217, 83, 79) 
        pdf.cell(0, 5, txt=akor_satiri, ln=True)
        pdf.set_text_color(44, 62, 80) 
        pdf.cell(0, 6, txt=soz_satiri, ln=True)
        if int(count) > 1:
            pdf.set_text_color(255, 75, 75)
            pdf.cell(0, 5, txt=f" (x{count})", ln=True, align='R')
        pdf.ln(4)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 3. SESSION STATE ---
# KALİBRASYON DEĞİŞKENİ ÇÖPE ATILDI
for k in ["analiz_verisi", "f_size", "orig_key", "target_key", "detected_bpm", "user_bpm", "metro_on"]:
    if k not in st.session_state:
        defaults = {"analiz_verisi":None, "f_size":22, "orig_key":"C", "target_key":"C", "detected_bpm":120, "user_bpm":120, "metro_on":False}
        st.session_state[k] = defaults[k]

# --- 4. ARAYÜZ (ALTIN KURAL: CENTERED VE KUTUSUZ TASARIM) ---
st.set_page_config(page_title="Studio Dashboard Pro", layout="centered")

st.markdown("""<style>
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    div.stButton > button { white-space: nowrap !important; padding: 4px 4px !important; font-size: 14px !important; border-radius: 6px !important; width: 100% !important; }
    div[data-testid="stHorizontalBlock"] { gap: 2px !important; }
    div[data-testid="column"] { padding-left: 1px !important; padding-right: 1px !important; }
    .lyrics-container { background: transparent !important; border: none !important; box-shadow: none !important; }
    
    /* AKOR ŞEMASI POPUP (TOOLTIP) */
    .chord-wrapper { position: relative; display: inline-block; cursor: pointer; }
    .chord-tooltip {
        visibility: hidden; background-color: #fff; border: 1px solid #ddd; border-radius: 8px;
        padding: 6px; position: absolute; z-index: 1000; bottom: 110%; left: 50%;
        transform: translateX(-50%); box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        opacity: 0; transition: opacity 0.2s, bottom 0.2s; pointer-events: none;
    }
    .chord-wrapper:hover .chord-tooltip, .chord-wrapper:active .chord-tooltip {
        visibility: visible; opacity: 1; bottom: 120%;
    }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    # MANUEL KALİBRASYON MENÜSÜ SİLİNDİ. Yan menü sadece söz düzenleme için belirecek.
    if st.session_state.analiz_verisi:
        st.markdown("### 📝 Sözleri Düzenle")
        mevcut = "\n".join([item[0] for item in st.session_state.analiz_verisi])
        yeni = st.text_area("", mevcut, height=500)
        if st.button("✅ Yansıt"):
            for i, satir in enumerate(yeni.split('\n')):
                if i < len(st.session_state.analiz_verisi):
                    st.session_state.analiz_verisi[i][0] = satir
                    for j, word in enumerate(satir.split()):
                        if j < len(st.session_state.analiz_verisi[i][1]): st.session_state.analiz_verisi[i][1][j]["k"] = word
            st.rerun()

st.markdown("<h1 style='text-align: center;'>🎸 Studio Dashboard Pro</h1>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1: url_input = st.text_input("YouTube linki:", placeholder="https://...")
with c2: yuklenen_dosya = st.file_uploader("Veya dosya:", type=["mp3", "wav"])

ca, cb_pdf, cb_bpm, c_minus, c_val, c_plus = st.columns([1.5, 1, 2, 0.5, 0.5, 0.5])

with ca: analiz_tetiklendi = st.button("🎵 Analiz", use_container_width=True)

with cb_pdf:
    if st.session_state.analiz_verisi:
        try:
            pdf_data = create_pdf(st.session_state.analiz_verisi, st.session_state.orig_key, st.session_state.target_key)
            st.download_button("📄 PDF", data=pdf_data, file_name="repertuvar.pdf", mime="application/pdf")
        except: st.button("📄 PDF Hata", disabled=True)
    else: st.button("📄 PDF", disabled=True)

with cb_bpm:
    st.session_state.user_bpm = st.slider("BPM", 40, 240, st.session_state.user_bpm, label_visibility="collapsed")
    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("▶" if not st.session_state.metro_on else "⏹"):
            st.session_state.metro_on = not st.session_state.metro_on; st.rerun()
    with c_b2:
        if st.button("🔄"): st.session_state.user_bpm = int(st.session_state.detected_bpm); st.rerun()

with c_minus: 
    if st.button("➖"): st.session_state.f_size = max(12, st.session_state.f_size - 2)
with c_val: 
    st.markdown(f"<div style='font-size:18px; text-align:center; font-weight:bold; line-height:34px; background-color: #f0f2f6; border-radius: 4px;'>{st.session_state.f_size}</div>", unsafe_allow_html=True)
with c_plus:
    if st.button("➕"): st.session_state.f_size = min(50, st.session_state.f_size + 2)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-weight: bold; margin-bottom: 5px; color: #555;'>🎼 Transpoze (Ton Değiştir)</p>", unsafe_allow_html=True)
sirali_notalar = (NOTALAR_STANDART[NOTALAR_STANDART.index(st.session_state.orig_key):] + NOTALAR_STANDART[:NOTALAR_STANDART.index(st.session_state.orig_key)]) if st.session_state.orig_key in NOTALAR_STANDART else NOTALAR_STANDART
t_cols = st.columns(len(sirali_notalar))
for i, nota in enumerate(sirali_notalar):
    with t_cols[i]:
        if st.button(nota, key=f"t_{nota}", use_container_width=True, type="primary" if st.session_state.target_key == nota else "secondary"):
            st.session_state.target_key = nota

# --- 5. ANALİZ SÜRECİ ---
if analiz_tetiklendi:
    source_title = ""
    with st.spinner("Gitar frekansları izole ediliyor..."):
        try:
            if url_input: source_title = download_audio_from_url(url_input)
            elif yuklenen_dosya:
                with open("temp_audio.mp3", "wb") as f: f.write(yuklenen_dosya.getbuffer())
                source_title = yuklenen_dosya.name

            y, sr = librosa.load("temp_audio.mp3")
            tempo_out, _ = librosa.beat.beat_track(y=y, sr=sr)
            st.session_state.detected_bpm = int(np.mean(tempo_out))
            st.session_state.user_bpm = st.session_state.detected_bpm
            
            model = get_fast_model()
            
            nyq = 0.5 * sr
            low = 150.0 / nyq
            high = 1000.0 / nyq
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            y_gitar = scipy.signal.filtfilt(b, a, y)
            
            y_harm = librosa.effects.hpss(y_gitar, margin=4.0)[0]
            
            # Oto-Akort kalibrasyonu devrede
            sapma = librosa.estimate_tuning(y=y_harm, sr=sr)
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=sapma, fmin=librosa.note_to_hz('E2'))
            chroma_temiz = scipy.signal.medfilt2d(chroma, kernel_size=(1, 7))
            
            segments, _ = model.transcribe(
                "temp_audio.mp3", 
                language="tr", 
                word_timestamps=True, 
                beam_size=5, 
                condition_on_previous_text=False,
                initial_prompt=f"Bu bir Türkçe şarkıdır. Şarkı: {source_title}"
            )
            
            all_words = []
            for s in segments:
                text_k = s.text.lower()
                if any(x in text_k for x in ["altyazı", "mustafa köse", "m.k", "m .k", "m k", "dimatorzok", "субтитры"]):
                    continue
                    
                for w in s.words:
                    kontrol = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', '', w.word).strip().lower()
                    if kontrol and kontrol not in KARA_LISTE: 
                        all_words.append(w)
            
            logical_rows, temp_row, last_end = [], [], 0
            for w in all_words:
                if (w.start - last_end > 1.2 or len(temp_row) >= 10) and temp_row:
                    logical_rows.append(temp_row); temp_row = []
                temp_row.append(w); last_end = w.end
            if temp_row: logical_rows.append(temp_row)
            
            final_sheet, son_akor, first_key = [], None, ""
            for row in logical_rows:
                row_content = []
                for i, word in enumerate(row):
                    idx = np.searchsorted(librosa.times_like(chroma_temiz, sr=sr), word.start)
                    avg = np.mean(chroma_temiz[:, idx:idx+3], axis=1)
                    
                    # MANUEL KAYDIRMA SİLİNDİ, saf normalize kullanılıyor
                    kalibre_avg_norm = avg / (np.linalg.norm(avg) + 1e-10)
                    
                    scores = np.dot(CHORD_TEMPLATES_NORM, kalibre_avg_norm)
                    best_idx = np.argmax(scores)
                    max_score = scores[best_idx]
                    
                    baskin = CHORD_LABELS[best_idx] if max_score > 0.55 else ""
                    
                    if not first_key and baskin: first_key = baskin.replace('m', '')
                    display = baskin if (baskin != son_akor and baskin != "") else ""
                    if baskin: son_akor = baskin
                    orijinal_kelime = word.word.strip()
                    if i == 0 and len(orijinal_kelime) > 0: orijinal_kelime = orijinal_kelime[0].upper() + orijinal_kelime[1:]
                    if orijinal_kelime: row_content.append({"a": display, "k": orijinal_kelime})
                
                if row_content:
                    row_txt = " ".join([x['k'] for x in row_content])
                    is_rep = False
                    
                    for j, old_item in enumerate(final_sheet):
                        if len(old_item) == 3 and difflib.SequenceMatcher(None, row_txt, old_item[0]).ratio() > 0.8:
                            final_sheet[j] = [old_item[0], old_item[1], old_item[2] + 1]
                            is_rep = True
                            break
                            
                    if not is_rep: final_sheet.append([row_txt, row_content, 1])

            st.session_state.analiz_verisi = final_sheet
            st.session_state.orig_key = first_key if first_key in INDEX_TO_KEY else "C"
            st.session_state.target_key = st.session_state.orig_key
            st.rerun()
        except Exception as e: st.error(f"Hata: {e}")

# --- 6. GÖRSELLEŞTİRME ---
if st.session_state.analiz_verisi:
    fs = st.session_state.f_size
    html_code = f'<div class="lyrics-container" style="font-family: sans-serif; margin-top: 10px;">'
    for item in st.session_state.analiz_verisi:
        if len(item) == 3: _, words, count = item
        else: _, words = item; count = 1
        html_code += f'<div style="display: flex; flex-wrap: wrap; align-items: flex-end; margin-bottom: {int(fs * 0.9)}px; padding-bottom: 4px;">'
        
        for w in words:
            t_chord = transpose_chord(w.get("a", ""), st.session_state.orig_key, st.session_state.target_key)
            
            if t_chord:
                svg_code = get_chord_svg(t_chord)
                if svg_code:
                    chord_h = f'''
                    <div class="chord-wrapper" style="color: #d9534f; font-weight: bold; font-family: monospace; font-size: {int(fs * 1.1)}px; min-height: 1.2em; text-align: center;">
                        {t_chord}
                        <div class="chord-tooltip">
                            <div style="font-size:11px; color:#333; margin-bottom:2px; font-family:sans-serif;">{t_chord} Akoru</div>
                            {svg_code}
                        </div>
                    </div>'''
                else:
                    chord_h = f'<div style="color: #d9534f; font-weight: bold; font-family: monospace; font-size: {int(fs * 1.1)}px; min-height: 1.2em; text-align: center;">{t_chord}</div>'
            else:
                chord_h = f'<div style="min-height: 1.2em;"></div>'
                
            html_code += f'<div style="display: inline-flex; flex-direction: column; align-items: center; margin-right: {int(fs * 0.6)}px; min-width: {int(fs * 1.4)}px;">{chord_h}<div style="color: #2c3e50; font-size: {fs}px; white-space: nowrap; font-weight: 500;">{w.get("k", "")}</div></div>'
            
        if int(count) > 1: html_code += f"<span style='color: #ff4b4b; font-size: 0.8em; font-weight: bold; margin-left: auto;'>x{count}</span>"
        html_code += "</div>"
    st.markdown(html_code + "</div>", unsafe_allow_html=True)