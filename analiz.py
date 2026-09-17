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
import streamlit.components.v1 as components

# --- 1. SABİTLER VE SİSTEM ---
NOTALAR_STANDART = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
MAP_12 = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11}
INDEX_TO_KEY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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
        svg += f'<line x1="{x}" y1="15" x2="{x}" y2="60" stroke="#8E8E93" stroke-width="1.5" stroke-linecap="round"/>'
    for i in range(5): 
        y = 15 + i * 11.25
        w = 2.5 if (i == 0 and start_fret == 1) else 1.5
        svg += f'<line x1="5" y1="{y}" x2="45" y2="{y}" stroke="#8E8E93" stroke-width="{w}" stroke-linecap="round"/>'

    if start_fret > 1: 
        svg += f'<text x="0" y="25" font-family="-apple-system, sans-serif" font-size="8" fill="#8E8E93">{start_fret}</text>'

    for i, fret in enumerate(frets): 
        x = 5 + i * 8
        if fret == 'x':
            svg += f'<text x="{x-3}" y="10" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#FF3B30">x</text>'
        elif fret == '0':
            svg += f'<circle cx="{x}" cy="10" r="2.5" fill="none" stroke="#8E8E93" stroke-width="1.5"/>'
        else:
            f = int(fret)
            rel_fret = f - start_fret
            y = 15 + rel_fret * 11.25 + 5.6
            svg += f'<circle cx="{x}" cy="{y}" r="4" fill="#FF3B30"/>'
    svg += '</svg>'
    return svg

def download_audio_from_url(url):
    os.environ["YTDLP_JS_RUNTIME"] = "node" 
    if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")
    
    # 403 Forbidden ve Bot Engellerini Aşmak İçin Güncellenen yt-dlp Ayarları
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': 'temp_audio', 
        'quiet': True, 
        'noplaylist': True, 
        'nocheckcertificate': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['webpage', 'configs']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_title = info.get('title', 'Bilinmeyen Şarkı')
        uploader = info.get('uploader', 'Bilinmeyen Sanatçı')
        artist = info.get('artist')
        track = info.get('track')

        if artist and track:
            return artist.strip(), track.strip()
        elif " - " in raw_title:
            parts = raw_title.split(" - ", 1)
            return parts[0].strip(), parts[1].strip()
        else:
            return uploader.replace(" - Topic", "").strip(), raw_title.strip()

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
            
        pdf.set_text_color(255, 59, 48)
        pdf.cell(0, 5, txt=akor_satiri, ln=True)
        pdf.set_text_color(28, 28, 30)
        pdf.cell(0, 6, txt=soz_satiri, ln=True)
        if int(count) > 1:
            pdf.set_text_color(142, 142, 147)
            pdf.cell(0, 5, txt=f" (x{count})", ln=True, align='R')
        pdf.ln(4)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 3. SESSION STATE VE ÇOKLU VERSİYONLU KÜTÜPHANE ---
for k in ["analiz_verisi", "f_size", "orig_key", "target_key", "detected_bpm", "user_bpm", "metro_on", "current_artist", "current_title", "current_version_idx", "view_mode", "global_song_db"]:
    if k not in st.session_state:
        defaults = {
            "analiz_verisi":None, "f_size":18, "orig_key":"C", "target_key":"C", 
            "detected_bpm":120, "user_bpm":120, "metro_on":False, 
            "current_artist": "", "current_title": "", "current_version_idx": 0, "view_mode": "home", 
            "global_song_db": [
                {
                    "artist": "Teoman", 
                    "title": "İki Yabancı", 
                    "versions": [
                        {
                            "name": "Versiyon 1 (Orijinal AI)",
                            "orig_key": "Am", 
                            "analiz_verisi": [["Sen ve ben", [{"a": "Am", "k": "Sen"}, {"a": "", "k": "ve"}, {"a": "C", "k": "ben"}], 1]]
                        }
                    ]
                }
            ]
        }
        st.session_state[k] = defaults[k]

def go_to_artist(artist_name):
    st.session_state.view_mode = "artist"
    st.session_state.current_artist = artist_name
    st.rerun()

def go_to_home():
    st.session_state.view_mode = "home"
    st.rerun()

def load_song_from_db(song_dict, version_idx=0):
    st.session_state.current_artist = song_dict['artist']
    st.session_state.current_title = song_dict['title']
    st.session_state.current_version_idx = version_idx
    
    active_version = song_dict['versions'][version_idx]
    st.session_state.analiz_verisi = active_version['analiz_verisi']
    st.session_state.orig_key = active_version['orig_key']
    st.session_state.target_key = active_version['orig_key']
    st.session_state.view_mode = "home"
    st.rerun()

def set_target_key(nota):
    st.session_state.target_key = nota
    st.rerun()

def fuzzy_search(query, song_db):
    if not query: return []
    query = query.lower().strip()
    results = []
    for s in song_db:
        text_to_search = f"{s['artist']} {s['title']}".lower()
        match_score = 0
        if query in text_to_search:
            match_score = 100
        else:
            words = text_to_search.split()
            q_words = query.split()
            for qw in q_words:
                matches = difflib.get_close_matches(qw, words, n=1, cutoff=0.6)
                if matches:
                    match_score += 50
        if match_score > 0:
            results.append((match_score, s))
    results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in results]

# --- 4. APPLE/macOS TASARIM (CSS ENJEKSİYONU - GÜNCELLENDİ) ---
st.set_page_config(page_title="ChordGenie Pro", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global Renk ve Arka Plan Sabitlemesi (Mobil Karanlık Mod Uyumlu) */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif !important;
    }
    
    .stApp {
        background-color: #F5F5F7 !important;
        color: #1D1D1F !important;
    }
    
    .block-container { 
        padding-top: 3rem !important; 
        padding-bottom: 2rem !important;
        max-width: 800px !important;
        background-color: #F5F5F7 !important;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Input Alanları */
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E5EA !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.2) !important;
    }
    
    /* Input İçindeki Yazı Rengi */
    input {
        color: #1D1D1F !important;
    }

    /* Buton Tasarımları */
    div.stButton > button { 
        background-color: #FFFFFF !important;
        color: #007AFF !important;
        border: 1px solid #E5E5EA !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
        padding: 6px 4px !important;
        white-space: nowrap !important;
    }
    
    div.stButton > button p {
        font-size: 13px !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }

    div.stButton > button:hover { background-color: #F2F2F7 !important; }
    div.stButton > button:active { transform: scale(0.97) !important; }
    
    div.stButton > button[kind="primary"] {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0, 122, 255, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #0066D6 !important; }

    div[data-testid="stHorizontalBlock"] { gap: 8px !important; }

    /* Şarkı Sözü ve Akor Konteyneri */
    .lyrics-container {
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04) !important;
        padding: 20px !important;
    }

    .chord-wrapper { 
        position: relative; 
        display: inline-block; 
        cursor: pointer; 
    }
    .chord-tooltip {
        visibility: hidden; 
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0,0,0,0.08); 
        border-radius: 16px;
        padding: 10px; 
        position: absolute; 
        z-index: 1000; 
        bottom: 130%; 
        left: 50%;
        transform: translateX(-50%) translateY(10px); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        opacity: 0; 
        transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1); 
        pointer-events: none;
    }
    .chord-wrapper:hover .chord-tooltip, .chord-wrapper:active .chord-tooltip {
        visibility: visible; 
        opacity: 1; 
        transform: translateX(-50%) translateY(0);
    }
    
    .lyric-text { color: #1D1D1F !important; font-weight: 500; letter-spacing: -0.2px; }
    .chord-text { color: #FF3B30 !important; font-weight: 700; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GÖRÜNÜM 1: SANATÇI SAYFASI
# ==========================================
if st.session_state.view_mode == "artist":
    if st.button("⬅ Geri Dön"):
        go_to_home()
        
    st.markdown(f"<h1 style='text-align: center; font-weight: 700; letter-spacing: -1px; margin-bottom: 5px;'>{st.session_state.current_artist}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8E8E93; font-size: 14px; margin-bottom: 30px;'>Ortak Repertuvar Kütüphanesi</p>", unsafe_allow_html=True)
    
    artist_songs = [s for s in st.session_state.global_song_db if s['artist'].lower() == st.session_state.current_artist.lower()]
    
    if artist_songs:
        for s in artist_songs:
            for idx, v in enumerate(s['versions']):
                label = f"🎵 {s['title']} — {v['name']} (Ton: {v['orig_key']})"
                if st.button(label, key=f"open_{s['title']}_{idx}", use_container_width=True):
                    load_song_from_db(s, idx)
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    else:
        st.info("Bu sanatçıya ait kayıtlı şarkı bulunamadı.")

# ==========================================
# GÖRÜNÜM 2: ANA SAYFA (DASHBOARD)
# ==========================================
elif st.session_state.view_mode == "home":
    
    st.markdown("<h2 style='text-align: center; font-weight: 800; letter-spacing: -1px; color: #1D1D1F; margin-bottom: 20px;'>ChordGenie<span style='color: #007AFF;'>.</span></h2>", unsafe_allow_html=True)

    col_search, col_url = st.columns(2)
    
    with col_search:
        search_query = st.text_input("Akıllı Arama", placeholder="Kütüphanede ara...", label_visibility="collapsed")
    
    with col_url:
        sub_c1, sub_c2 = st.columns([3, 1])
        with sub_c1:
            url_input = st.text_input("Bağlantı", placeholder="YouTube URL...", label_visibility="collapsed")
        with sub_c2:
            analiz_tetiklendi = st.button("Analiz", use_container_width=True, type="primary")

    if search_query:
        search_results = fuzzy_search(search_query, st.session_state.global_song_db)
        if search_results:
            st.markdown("<p style='font-size: 13px; color: #8E8E93; margin-top: 10px;'>Kütüphanede bulunan eşleşmeler:</p>", unsafe_allow_html=True)
            for res in search_results:
                for idx, v in enumerate(res['versions']):
                    if st.button(f"✨ {res['artist']} - {res['title']} [{v['name']}] (Ton: {v['orig_key']})", key=f"search_{res['title']}_{idx}", use_container_width=True):
                        load_song_from_db(res, idx)
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #E5E5EA;'>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 13px; color: #FF3B30; margin-top: 10px;'>Kütüphanede eşleşen şarkı bulunamadı.</p>", unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if st.session_state.analiz_verisi:
        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb, cc, cd, ce = st.columns([1, 1, 1, 1, 1])
        with ca:
            if st.button("📥 PDF İndir", use_container_width=True):
                pass
        with cb:
            if st.button("⏱ Metronom", use_container_width=True, type="primary" if st.session_state.metro_on else "secondary"):
                st.session_state.metro_on = not st.session_state.metro_on
                st.rerun()
        with cc:
             st.markdown(f"<div style='text-align:center; font-weight:600; line-height:36px; color:#007AFF;' id='bpm-display'>{st.session_state.user_bpm} BPM</div>", unsafe_allow_html=True)
        with cd: 
            if st.button("➖ A-", use_container_width=True): st.session_state.f_size = max(12, st.session_state.f_size - 2)
        with ce:
            if st.button("➕ A+", use_container_width=True): st.session_state.f_size = min(50, st.session_state.f_size + 2)

        current_bpm = st.session_state.user_bpm
        detected_bpm = st.session_state.detected_bpm
        
        slider_html = f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 14px; padding: 12px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); display: flex; align-items: center; justify-content: space-between; gap: 12px; font-family: -apple-system, sans-serif; margin-top: 10px;">
            <button onclick="changeBPM(-5)" style="background: #F2F2F7; border: none; border-radius: 8px; padding: 6px 10px; font-weight: 600; color: #1D1D1F; cursor: pointer; font-size: 12px;">-5</button>
            <input type="range" id="bpmRange" min="40" max="240" value="{current_bpm}" style="flex-grow: 1; accent-color: #007AFF; cursor: pointer;" oninput="updateBPM(this.value)">
            <button onclick="changeBPM(5)" style="background: #F2F2F7; border: none; border-radius: 8px; padding: 6px 10px; font-weight: 600; color: #1D1D1F; cursor: pointer; font-size: 12px;">+5</button>
            <button onclick="resetBPM({detected_bpm})" style="background: #EAF2FF; color: #007AFF; border: none; border-radius: 8px; padding: 6px 12px; font-weight: 600; cursor: pointer; font-size: 12px;" title="Orijinal Hıza Dön">Orijinal ({detected_bpm})</button>
        </div>
        <script>
            function updateBPM(val) {{
                parent.window.currentBPM = val;
                const disp = window.parent.document.getElementById('bpm-display');
                if(disp) disp.innerText = val + " BPM";
                
                if (window.parent.metronomInterval) {{
                    window.parent.restartMetronom(val);
                }}
            }}
            function changeBPM(amount) {{
                const slider = document.getElementById('bpmRange');
                let newVal = parseInt(slider.value) + amount;
                if(newVal < 40) newVal = 40;
                if(newVal > 240) newVal = 240;
                slider.value = newVal;
                updateBPM(newVal);
            }}
            function resetBPM(origVal) {{
                const slider = document.getElementById('bpmRange');
                slider.value = origVal;
                updateBPM(origVal);
            }}
        </script>
        """
        components.html(slider_html, height=70)

        if st.session_state.metro_on:
            bpm = st.session_state.user_bpm
            metronom_html = f"""
            <div style="text-align:center; padding: 2px; font-family: -apple-system, sans-serif; font-size: 11px; color: #007AFF; font-weight: 600;">
                🟢 Metronom Aktif
            </div>
            <script>
                window.parent.currentBPM = {bpm};
                
                function playClick() {{
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = audioCtx.createOscillator();
                    const envelope = audioCtx.createGain();
                    
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(800, audioCtx.currentTime);
                    
                    envelope.gain.setValueAtTime(1, audioCtx.currentTime);
                    envelope.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.05);
                    
                    osc.connect(envelope);
                    envelope.connect(audioCtx.destination);
                    
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.05);
                }}
                
                window.parent.restartMetronom = function(newBpm) {{
                    if (window.parent.metronomInterval) {{
                        clearInterval(window.parent.metronomInterval);
                    }}
                    const interval = (60 / newBpm) * 1000;
                    window.parent.metronomInterval = setInterval(playClick, interval);
                }};
                
                window.parent.restartMetronom(window.parent.currentBPM);
            </script>
            """
            components.html(metronom_html, height=30)
        else:
            stop_html = """
            <script>
                if (window.parent.metronomInterval) {
                    clearInterval(window.parent.metronomInterval);
                    window.parent.metronomInterval = null;
                }
            </script>
            """
            components.html(stop_html, height=0)

        st.markdown("<p style='text-align: center; font-weight: 600; font-size: 12px; margin-top: 15px; margin-bottom: 5px; color: #8E8E93; text-transform: uppercase; letter-spacing: 1px;'>Transpoze</p>", unsafe_allow_html=True)
        
        sirali_notalar = (NOTALAR_STANDART[NOTALAR_STANDART.index(st.session_state.orig_key):] + NOTALAR_STANDART[:NOTALAR_STANDART.index(st.session_state.orig_key)]) if st.session_state.orig_key in NOTALAR_STANDART else NOTALAR_STANDART
        
        t_cols = st.columns(len(sirali_notalar))
        for i, nota in enumerate(sirali_notalar):
            with t_cols[i]:
                is_active = (st.session_state.target_key == nota)
                if st.button(nota, key=f"t_{nota}", use_container_width=True, type="primary" if is_active else "secondary"):
                    set_target_key(nota)

    if analiz_tetiklendi:
        with st.spinner("Yapay zeka ortak kütüphane için işliyor..."):
            try:
                artist_name, track_name = "Bilinmeyen Sanatçı", "Bilinmeyen Şarkı"
                if url_input: 
                    artist_name, track_name = download_audio_from_url(url_input)

                y, sr = librosa.load("temp_audio.mp3")
                tempo_out, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                st.session_state.detected_bpm = int(np.mean(tempo_out))
                st.session_state.user_bpm = st.session_state.detected_bpm
                
                model = get_fast_model()
                
                nyq = 0.5 * sr
                low = 150.0 / nyq
                high = 1000.0 / nyq
                b, a = scipy.signal.butter(4, [low, high], btype='band')
                y_gitar = scipy.signal.filtfilt(b, a, y)
                y_harm = librosa.effects.hpss(y_gitar, margin=4.0)[0]
                
                sapma = librosa.estimate_tuning(y=y_harm, sr=sr)
                chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=sapma, fmin=librosa.note_to_hz('E2'))
                chroma_temiz = scipy.signal.medfilt2d(chroma, kernel_size=(1, 15))
                
                segments, _ = model.transcribe(
                    "temp_audio.mp3", 
                    language=None, 
                    word_timestamps=True, 
                    beam_size=5, 
                    condition_on_previous_text=False,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
                    vad_filter=False
                )
                
                all_words = []
                for s in segments:
                    for w in s.words:
                        clean_w = w.word.strip()
                        kontrol = re.sub(r'[^a-zA-Z0-9\sğüşıöçĞÜŞİÖÇ\'\’]', '', clean_w).strip()
                        
                        if len(kontrol) > 0: 
                            all_words.append({"start": w.start, "end": w.end, "word": kontrol})
                
                logical_rows, temp_row, last_end = [], [], 0
                for w in all_words:
                    if (w["start"] - last_end > 2.0 or len(temp_row) >= 12) and temp_row:
                        logical_rows.append(temp_row); temp_row = []
                    temp_row.append(w); last_end = w["end"]
                if temp_row: logical_rows.append(temp_row)
                
                final_sheet, son_akor, first_key = [], None, ""
                for row in logical_rows:
                    row_content = []
                    for i, word in enumerate(row):
                        idx = np.searchsorted(librosa.times_like(chroma_temiz, sr=sr), word["start"])
                        avg = np.mean(chroma_temiz[:, idx:idx+5], axis=1)
                        kalibre_avg_norm = avg / (np.linalg.norm(avg) + 1e-10)
                        
                        scores = np.dot(CHORD_TEMPLATES_NORM, kalibre_avg_norm)
                        best_idx = np.argmax(scores)
                        max_score = scores[best_idx]
                        
                        baskin = CHORD_LABELS[best_idx] if max_score > 0.62 else ""
                        
                        if not first_key and baskin: first_key = baskin.replace('m', '')
                        
                        display = ""
                        if baskin and baskin != son_akor:
                            display = baskin
                            son_akor = baskin
                        
                        orijinal_kelime = word["word"]
                        if i == 0 and len(orijinal_kelime) > 0: orijinal_kelime = orijinal_kelime[0].upper() + orijinal_kelime[1:]
                        if orijinal_kelime: row_content.append({"a": display, "k": orijinal_kelime})
                    
                    if row_content:
                        row_txt = " ".join([x['k'] for x in row_content])
                        is_rep = False
                        for j, old_item in enumerate(final_sheet):
                            if len(old_item) == 3 and difflib.SequenceMatcher(None, row_txt, old_item[0]).ratio() > 0.8:
                                final_sheet[j] = [old_item[0], old_item[1], old_item[2] + 1]; is_rep = True; break
                        if not is_rep: final_sheet.append([row_txt, row_content, 1])

                detected_key = first_key if first_key in INDEX_TO_KEY else "C"
                
                existing_song = next((s for s in st.session_state.global_song_db if s['title'].lower() == track_name.lower() and s['artist'].lower() == artist_name.lower()), None)
                
                new_version_data = {
                    "name": f"Versiyon (AI Analiz)",
                    "orig_key": detected_key,
                    "analiz_verisi": final_sheet
                }

                if existing_song:
                    new_version_data["name"] = f"Versiyon {len(existing_song['versions']) + 1}"
                    existing_song['versions'].append(new_version_data)
                    target_song = existing_song
                    target_v_idx = len(existing_song['versions']) - 1
                else:
                    new_song_entry = {
                        "artist": artist_name,
                        "title": track_name,
                        "versions": [new_version_data]
                    }
                    st.session_state.global_song_db.append(new_song_entry)
                    target_song = new_song_entry
                    target_v_idx = 0

                load_song_from_db(target_song, target_v_idx)

            except Exception as e: st.error(f"Bir sorun oluştu: {e}")

    # --- 6. GÖRSELLEŞTİRME, VERSİYON SEÇİCİ VE BAŞLIK ---
    if st.session_state.analiz_verisi:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-weight: 800; font-size: 32px; letter-spacing: -1px; margin-bottom: 5px;'>{st.session_state.current_title}</h1>", unsafe_allow_html=True)
        
        current_song_obj = next((s for s in st.session_state.global_song_db if s['title'] == st.session_state.current_title and s['artist'] == st.session_state.current_artist), None)
        
        if current_song_obj and len(current_song_obj['versions']) > 1:
            version_names = [v['name'] for v in current_song_obj['versions']]
            selected_v_name = st.selectbox("Şarkı Versiyonu Seç", version_names, index=st.session_state.current_version_idx, label_visibility="collapsed")
            selected_idx = version_names.index(selected_v_name)
            if selected_idx != st.session_state.current_version_idx:
                load_song_from_db(current_song_obj, selected_idx)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"👤 {st.session_state.current_artist}", use_container_width=True):
                go_to_artist(st.session_state.current_artist)
        
        st.markdown("<br>", unsafe_allow_html=True)

        fs = st.session_state.f_size
        html_code = f'<div class="lyrics-container" style="margin-top: 20px; padding: 20px; background-color: #FFFFFF; border-radius: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.04);">'
        for item in st.session_state.analiz_verisi:
            if len(item) == 3: _, words, count = item
            else: _, words = item; count = 1
            html_code += f'<div style="display: flex; flex-wrap: wrap; align-items: flex-end; margin-bottom: {int(fs * 1.2)}px; padding-bottom: 4px;">'
            
            for w in words:
                t_chord = transpose_chord(w.get("a", ""), st.session_state.orig_key, st.session_state.target_key)
                
                if t_chord:
                    svg_code = get_chord_svg(t_chord)
                    if svg_code:
                        chord_h = f'''
                        <div class="chord-wrapper chord-text" style="font-size: {int(fs * 0.9)}px; min-height: 1.4em; text-align: center; margin-bottom: 2px;">
                            {t_chord}
                            <div class="chord-tooltip">
                                <div style="font-size:12px; color:#1D1D1F; font-weight: 600; margin-bottom:6px; font-family:sans-serif;">{t_chord} Akoru</div>
                                {svg_code}
                            </div>
                        </div>'''
                    else:
                        chord_h = f'<div class="chord-text" style="font-size: {int(fs * 0.9)}px; min-height: 1.4em; text-align: center; margin-bottom: 2px;">{t_chord}</div>'
                else:
                    chord_h = f'<div style="min-height: 1.4em; margin-bottom: 2px;"></div>'
                    
                html_code += f'<div style="display: inline-flex; flex-direction: column; align-items: center; margin-right: {int(fs * 0.5)}px; min-width: {int(fs * 1.3)}px;">{chord_h}<div class="lyric-text" style="font-size: {fs}px; white-space: nowrap;">{w.get("k", "")}</div></div>'
                
            if int(count) > 1: html_code += f"<span style='color: #8E8E93; font-size: {int(fs * 0.8)}px; font-weight: 600; margin-left: auto; background: #F2F2F7; padding: 2px 8px; border-radius: 12px;'>x{count}</span>"
            html_code += "</div>"
        st.markdown(html_code + "</div>", unsafe_allow_html=True)
