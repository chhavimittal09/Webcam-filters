"""
Lumina Studio - Real-Time Webcam Filters - Streamlit Web App
=================================================================
Live webcam feed, 10 filters, filter combos, adjustable intensity,
mirror toggle, keyboard-only control, snapshot + video capture,
3-shot photo strip, countdown photobooth mode, zip download, and an
in-browser gallery. Full-width, clean "bento box" layout where each
panel has its own solid signature color and the accent reacts to
whichever filter is active.

INSTALL (once):
    pip install streamlit streamlit-webrtc av opencv-python-headless numpy

RUN:
    streamlit run webcamfilter.py

Open the printed URL (usually http://localhost:8501), click Start under
the video, and approve the camera permission prompt.

IF THE BROWSER SAYS "Camera not allowed":
    That is a browser permission, not a code bug. Click the camera icon in
    the address bar, set Camera to Allow, then reload the page.

IF THE PREVIEW STAYS BLANK EVEN AFTER ALLOWING:
    A webcam can only be held by one app at a time. Close Zoom, Teams, the
    Windows Camera app, or any other tab using the camera, then reload.
    On Windows, also check Settings > Privacy & security > Camera > make
    sure "Let apps access your camera" and "Let desktop apps access your
    camera" are both turned on.

KEYBOARD SHORTCUTS (work anywhere on the page, no clicking needed):
    1-9   switch to filters 1-9
    0     switch to the 10th filter (Signature Duotone)
    s     save current frame to the in-app gallery
    c     start a 3-2-1 countdown, then save (photobooth mode)
    p     capture a 3-shot photo strip
    r     start / stop video recording
"""

import io
import os
import glob
import random
import zipfile
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

# --------------------------------------------------------------------------- #
# Filter implementations
# --------------------------------------------------------------------------- #

def apply_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64)
    sepia = img_rgb @ kernel.T
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)


def _vignette_mask(h, w, strength=2.2):
    kernel_x = cv2.getGaussianKernel(w, w / strength)
    kernel_y = cv2.getGaussianKernel(h, h / strength)
    kernel = kernel_y * kernel_x.T
    return kernel / kernel.max()


def apply_vintage(frame):
    sepia = apply_sepia(frame)
    h, w = sepia.shape[:2]
    mask = _vignette_mask(h, w)
    vignette = sepia.astype(np.float64)
    for c in range(3):
        vignette[:, :, c] *= mask
    vignette = np.clip(vignette, 0, 255).astype(np.uint8)
    noise = np.random.randint(-18, 18, vignette.shape, dtype=np.int16)
    return np.clip(vignette.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_cartoon(frame):
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=200, sigmaSpace=200)
    color = cv2.bilateralFilter(color, d=9, sigmaColor=200, sigmaSpace=200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=2
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_colored)


def _build_lut(shift):
    return np.clip(np.arange(0, 256) + shift, 0, 255).astype(np.uint8)


_WARM_R_LUT = _build_lut(30)
_WARM_B_LUT = _build_lut(-30)
_COOL_R_LUT = _build_lut(-30)
_COOL_B_LUT = _build_lut(30)


def apply_warm(frame):
    b, g, r = cv2.split(frame)
    r = cv2.LUT(r, _WARM_R_LUT)
    b = cv2.LUT(b, _WARM_B_LUT)
    return cv2.merge((b, g, r))


def apply_cool(frame):
    b, g, r = cv2.split(frame)
    r = cv2.LUT(r, _COOL_R_LUT)
    b = cv2.LUT(b, _COOL_B_LUT)
    return cv2.merge((b, g, r))


def apply_neon_glow(frame):
    """Custom bonus filter: neon edge glow on a dark background.

    cv2.applyColorMap maps 0 to a real color, not black, so a naive version
    colors the whole frame instead of just the edges. The colored edges are
    masked by the edge map itself before blurring, so only edges glow.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    edges_color = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
    edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    edges_color = (edges_color.astype(np.float32) * edge_mask).astype(np.uint8)

    glow = cv2.GaussianBlur(edges_color, (0, 0), sigmaX=6)
    dark_bg = (frame * 0.15).astype(np.uint8)
    return cv2.addWeighted(dark_bg, 1.0, glow, 1.4, 0)


# Custom bonus filter: maps brightness onto the app's own brand gradient
# (near-black to indigo) instead of black to white.
_DUOTONE_DARK = np.array([10, 5, 5], dtype=np.float32)     # BGR near-black
_DUOTONE_LIGHT = np.array([252, 94, 109], dtype=np.float32)  # BGR indigo/violet


def apply_signature_duotone(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    duo = _DUOTONE_DARK[None, None, :] + gray[..., None] * (_DUOTONE_LIGHT - _DUOTONE_DARK)[None, None, :]
    return np.clip(duo, 0, 255).astype(np.uint8)


FILTER_ORDER = [
    "Original", "Grayscale", "Sepia", "Vintage", "Sketch",
    "Cartoon", "Warm Tone", "Cool Tone", "Neon Glow", "Signature Duotone",
]
FILTER_FUNCS = {
    "Original": lambda f: f,
    "Grayscale": apply_grayscale,
    "Sepia": apply_sepia,
    "Vintage": apply_vintage,
    "Sketch": apply_sketch,
    "Cartoon": apply_cartoon,
    "Warm Tone": apply_warm,
    "Cool Tone": apply_cool,
    "Neon Glow": apply_neon_glow,
    "Signature Duotone": apply_signature_duotone,
}
FILTER_ICONS = {
    "Original": "\U0001F3AC", "Grayscale": "\u2B1B", "Sepia": "\U0001F7E4",
    "Vintage": "\U0001F4FB", "Sketch": "\u270F\uFE0F", "Cartoon": "\U0001F3A8",
    "Warm Tone": "\U0001F525", "Cool Tone": "\u2744\uFE0F", "Neon Glow": "\U0001F4A1",
    "Signature Duotone": "\U0001F30C",
}

# Each filter gets its own signature color, used to color the active-filter
# card and the nav dot so the interface visibly reacts to the creative
# choice you just made instead of staying one flat color forever.
FILTER_COLORS = {
    "Original": "#6b7280",
    "Grayscale": "#4b5563",
    "Sepia": "#b45309",
    "Vintage": "#c2410c",
    "Sketch": "#0284c7",
    "Cartoon": "#db2777",
    "Warm Tone": "#e11d48",
    "Cool Tone": "#0891b2",
    "Neon Glow": "#9333ea",
    "Signature Duotone": "#7c3aed",
}

# BGR versions (for drawing divider bars on captured frames with cv2)
def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

IMAGE_EXTS = (".png",)
VIDEO_EXTS = (".mp4",)

# --------------------------------------------------------------------------- #
# Page setup + theme: clean, flat "bento box" UI. Every panel gets its own
# solid signature-colored background (no blur, no animated blobs) so the
# page reads as a set of distinct colorful cards rather than a hazy glass
# effect layered over everything.
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Lumina Studio", page_icon="\u25C9", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg:        #f5f5fa;
    --surface:   #ffffff;
    --surface-2: #f0eff8;
    --border:    #e3e1ee;
    --text:      #17131f;
    --muted:     #6d6980;
    --good:      #16a34a;
    --accent-live: #7c3aed;

    --panel-camera:   14, 165, 233;
    --panel-combo:    217, 70, 239;
    --panel-filters:  245, 158, 11;
    --panel-capture:  16, 185, 129;
    --panel-active:   124, 58, 237;
}

html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.block-container { max-width: 1360px !important; padding: 1.2rem 2.4rem 3rem 2.4rem !important; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }

.st-key-nav_bar div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border) !important; border-radius: 16px;
    padding: 0.6rem 1.2rem !important;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(20,10,40,0.05);
}
.st-key-nav_bar div[data-testid="stHorizontalBlock"] { align-items: center; }

/* ---------- Panel background colors (real CSS, not JS) ----------
   Each rule is written twice on purpose: once assuming Streamlit puts the
   st-key-<name> class on the SAME element as the bordered-container
   testid, once assuming it's on a PARENT of it. Only one of the two will
   ever actually match in a given Streamlit version, but writing both
   means it can't silently miss the way the old JS .closest() lookup did -
   whichever guess is right just works, no debugging required later. */
.st-key-panel_camera[data-testid="stVerticalBlockBorderWrapper"],
.st-key-panel_camera div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(250, 220, 30, 0.28) !important;
    border: none !important;
    border-left: 6px solid rgb(202, 138, 4) !important;
    border-radius: 16px !important;
}
.st-key-panel_combo[data-testid="stVerticalBlockBorderWrapper"],
.st-key-panel_combo div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(168, 85, 247, 0.22) !important;
    border: none !important;
    border-left: 6px solid rgb(126, 34, 206) !important;
    border-radius: 16px !important;
}
.st-key-panel_filters[data-testid="stVerticalBlockBorderWrapper"],
.st-key-panel_filters div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(249, 168, 212, 0.32) !important;
    border: none !important;
    border-left: 6px solid rgb(219, 39, 119) !important;
    border-radius: 16px !important;
}
.st-key-panel_capture[data-testid="stVerticalBlockBorderWrapper"],
.st-key-panel_capture div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(56, 189, 248, 0.24) !important;
    border: none !important;
    border-left: 6px solid rgb(2, 132, 199) !important;
    border-radius: 16px !important;
}
.st-key-panel_gallery[data-testid="stVerticalBlockBorderWrapper"],
.st-key-panel_gallery div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(251, 146, 60, 0.24) !important;
    border: none !important;
    border-left: 6px solid rgb(194, 65, 12) !important;
    border-radius: 16px !important;
}
.st-key-theme_toggle_btn button {
    border-radius: 999px !important; width: 2.5rem; padding: 0 !important;
    font-size: 1.1rem;
}
.nav-brand {
    display: flex; align-items: center; gap: 0.7rem; font-family: 'Sora', sans-serif;
    font-weight: 800; font-size: 1.15rem; color: var(--text);
}
.nav-brand .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent-live);
    transition: background 0.35s ease;
}
.status-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.35rem 0.9rem; font-size: 0.82rem; color: var(--muted);
}
.status-pill .led { width: 8px; height: 8px; border-radius: 50%; }
.led-on { background: var(--good); }
.led-rec { background: #e11d48; animation: recPulse 1.1s ease-in-out infinite; }
.led-off { background: #c7c4d6; }
@keyframes recPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

.hero { padding: 0.6rem 0.1rem 1.4rem 0.1rem; }
.hero h1 { font-size: 2.2rem; margin: 0; color: var(--text); }
.hero h1 .accent-word { color: var(--accent-live); transition: color 0.35s ease; }
.hero p { color: var(--muted); font-size: 0.98rem; margin: 0.5rem 0 0 0; max-width: 640px; }

div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 0.7rem 1rem; box-shadow: 0 2px 8px rgba(20,10,40,0.04);
}
div[data-testid="stMetricValue"] { color: var(--accent-live); font-family: 'Sora', sans-serif; transition: color 0.35s ease; }
div[data-testid="stMetricLabel"] { color: var(--muted); }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border) !important; border-radius: 16px;
    padding: 0.9rem 1rem; margin-bottom: 1.1rem;
    box-shadow: 0 3px 12px rgba(20,10,40,0.05);
}

/* Panel and metric colors are applied via JavaScript (see the script
   below) rather than CSS, because Streamlit's container DOM nesting
   made every pure-CSS selector approach (key-based classes, marker
   siblings, :has()) unreliable across versions. Direct DOM styling
   via element.closest() doesn't depend on guessing that structure. */

/* Green transparent card for the hero section */
.hero-card {
    background: rgba(16, 185, 129, 0.16);
    border-left: 6px solid #10b981;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
}

/* Stat metric colors are applied via the same JS approach below. */

.panel-title { color: #000000 !important; }
.panel-title {
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1rem;
    margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.4rem;
}
.panel-hint { color: var(--muted); font-size: 0.78rem; margin: 0 0 0.8rem 0; }

.active-filter-card {
    border: none;
    border-left: 6px solid rgb(var(--panel-active));
    background: rgba(var(--panel-active), 0.24);
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 1.1rem;
    transition: background 0.3s ease;
}
.active-filter-card .eyebrow { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }
.active-filter-card .value {
    font-size: 1.3rem; font-weight: 800; color: var(--accent-live); margin-top: 0.2rem;
    font-family: 'Sora', sans-serif; transition: color 0.35s ease;
}
.active-filter-card .combo-note { color: var(--muted); font-size: 0.78rem; margin-top: 0.3rem; }

div[data-testid="stVideoFrame"], video {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

.stButton>button, .stDownloadButton>button {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 10px; font-weight: 600; font-size: 0.84rem; height: 2.5rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: border-color 0.12s ease, transform 0.08s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--accent-live); color: var(--accent-live); transform: translateY(-1px);
}
.stButton>button[kind="primary"] {
    background: var(--accent-live);
    color: #ffffff; border: none; font-weight: 800;
}
.stButton>button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }

.kbd {
    display: inline-block; background: var(--surface-2); border: 1px solid var(--accent-live);
    color: var(--accent-live); border-radius: 6px; padding: 1px 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; margin-right: 3px;
    transition: color 0.35s ease, border-color 0.35s ease;
}

div[data-testid="stCheckbox"] label p { color: var(--text); }
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--accent-live) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent-live) !important; }

.gallery-card {
    border: none; border-radius: 12px;
    padding: 0.6rem; margin-bottom: 0.9rem;
    transition: transform 0.1s ease;
    box-shadow: 0 2px 8px rgba(20,10,40,0.04);
}
.gallery-card:hover { transform: translateY(-2px); }
.gallery-card.kind-photo { background: rgba(124, 58, 237, 0.20); border-left: 6px solid #7c3aed; }
.gallery-card.kind-video { background: rgba(225, 29, 72, 0.20); border-left: 6px solid #e11d48; }
.gallery-card.kind-strip { background: rgba(8, 145, 178, 0.20); border-left: 6px solid #0891b2; }
.gallery-caption { color: var(--muted); font-size: 0.74rem; text-align: center; margin-top: 0.4rem; word-break: break-all; }
.gallery-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 1px 7px; border-radius: 999px; margin-bottom: 0.35rem;
}
.badge-photo { background: #ede9fe; color: #7c3aed; }
.badge-video { background: #ffe4e6; color: #e11d48; }
.badge-strip { background: #cffafe; color: #0891b2; }

.empty-state {
    border: 1px dashed var(--border); border-radius: 16px; padding: 1.8rem;
    text-align: center; color: var(--muted); font-size: 0.9rem;
    background: transparent;
}

.st-key-panel_gallery h3 { color: #000000 !important; }

.site-footer {
    margin-top: 2rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 0.8rem; text-align: center;
}

hr { border-color: var(--border); }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent-live); border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

_defaults = {
    "filter_name": "Original",
    "pending_save": False,
    "pending_strip": False,
    "last_shown_capture": None,
    "session_start": datetime.now(),
    "mirror": True,
    "intensity": 100,
    "combo_enabled": False,
    "combo_filter": "Neon Glow",
    "combo_intensity": 50,
    "recording": False,
    "theme": "light",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

_active_color = FILTER_COLORS.get(st.session_state.filter_name, "#8b5cf6")
st.markdown(f"<style>:root {{ --accent-live: {_active_color}; }}</style>", unsafe_allow_html=True)

if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    :root {
        --bg:        #0b0a10;
        --surface:   #15131e;
        --surface-2: #1b1927;
        --border:    #2b2840;
        --text:      #f1eff8;
        --muted:     #a29dbd;
        --good:      #34d399;
    }
    </style>
    """, unsafe_allow_html=True)

is_dark = st.session_state.theme == "dark"

# --------------------------------------------------------------------------- #
# Keyboard-only control + countdown photobooth overlay, built from scratch.
# Every filter/save/record button below is labeled with a bracketed key,
# e.g. "[1] Original" or "[S] Save current frame". This script finds and
# directly clicks the matching real Streamlit button whenever that key is
# pressed - the same thing your finger clicking it would do.
# --------------------------------------------------------------------------- #

components.html("""
<script>
(function() {
    function clickByPrefix(doc, prefix) {
        var buttons = doc.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var t = (buttons[i].innerText || '').trim();
            if (t.indexOf(prefix) === 0) {
                buttons[i].click();
                return true;
            }
        }
        return false;
    }

    function runCountdown(doc) {
        var existing = doc.getElementById('lumina-countdown-overlay');
        if (existing) return;
        var el = doc.createElement('div');
        el.id = 'lumina-countdown-overlay';
        el.style.cssText = [
            'position:fixed', 'inset:0', 'display:flex', 'align-items:center',
            'justify-content:center', 'font-size:14vw', 'font-weight:800',
            'font-family:Sora,sans-serif', 'color:#fff',
            'background:rgba(6,6,15,0.55)', 'backdrop-filter:blur(6px)',
            'z-index:999999', 'pointer-events:none',
            'text-shadow:0 0 40px #8b5cf6, 0 0 90px #ec4899'
        ].join(';');
        doc.body.appendChild(el);
        var n = 3;
        el.innerText = n;
        var iv = setInterval(function() {
            n -= 1;
            if (n <= 0) {
                clearInterval(iv);
                el.innerText = '\u2728';
                setTimeout(function() {
                    clickByPrefix(doc, '[S]');
                    el.remove();
                }, 250);
            } else {
                el.innerText = n;
            }
        }, 800);
    }

    function onKey(e) {
        if (e.ctrlKey || e.altKey || e.metaKey) return;
        var tag = (e.target && e.target.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        var key = e.key;
        var doc = window.parent.document;
        if (/^[0-9]$/.test(key)) {
            clickByPrefix(doc, '[' + key + ']');
        } else if (key === 's' || key === 'S') {
            clickByPrefix(doc, '[S]');
        } else if (key === 'c' || key === 'C') {
            runCountdown(doc);
        } else if (key === 'p' || key === 'P') {
            clickByPrefix(doc, '[P]');
        } else if (key === 'r' || key === 'R') {
            clickByPrefix(doc, '[R]');
        }
    }

    function onClick(e) {
        var el = e.target;
        for (var depth = 0; el && depth < 4; depth++, el = el.parentElement) {
            if (el.tagName === 'BUTTON') {
                var t = (el.innerText || '').trim();
                if (t.indexOf('\u23F3 Countdown') === 0) {
                    runCountdown(window.parent.document);
                }
                break;
            }
        }
    }

    function hookDoc(doc) {
        try {
            if (doc && !doc.__luminaHooked) {
                doc.addEventListener('keydown', onKey, true);
                doc.addEventListener('click', onClick, true);
                doc.__luminaHooked = true;
            }
        } catch (e) {}
    }

    function hookAll() {
        try {
            var topDoc = window.parent.document;
            hookDoc(topDoc);
            var iframes = topDoc.querySelectorAll('iframe');
            for (var i = 0; i < iframes.length; i++) {
                try { hookDoc(iframes[i].contentDocument); } catch (e) {}
            }
        } catch (e) {}
    }

    function reclaimFocus() {
        try {
            var doc = window.parent.document;
            var active = doc.activeElement;
            if (active && active.tagName === 'IFRAME') {
                active.blur();
                if (!doc.body.hasAttribute('tabindex')) {
                    doc.body.setAttribute('tabindex', '-1');
                    doc.body.style.outline = 'none';
                }
                doc.body.focus({preventScroll: true});
            }
        } catch (e) {}
    }

    function colorizeMetrics(doc) {
        // The 5 panels (Camera/Combo/Filters/Capture/Saved Shots) are now
        // colored with plain CSS above - it doesn't depend on a runtime
        // DOM lookup, so it can't silently miss the way this script's
        // .closest()-based panel coloring used to. Only the stat-metric
        // row is still colored here, since st.metric() doesn't expose a
        // `key=` to hook a CSS class onto.
        var metricColors = ['rgba(14,165,233,0.20)', 'rgba(124,58,237,0.20)',
                             'rgba(245,158,11,0.20)', 'rgba(225,29,72,0.20)',
                             'rgba(16,185,129,0.20)'];
        var metricBorders = ['rgb(2,132,199)', 'rgb(109,40,217)',
                              'rgb(180,83,9)', 'rgb(190,18,60)', 'rgb(4,120,87)'];
        var metrics = doc.querySelectorAll('[data-testid="stMetric"]');
        metrics.forEach(function(m, i) {
            m.style.setProperty('background', metricColors[i % 5], 'important');
            m.style.setProperty('border', 'none', 'important');
            m.style.setProperty('border-left', '4px solid ' + metricBorders[i % 5], 'important');
            m.style.setProperty('border-radius', '12px', 'important');
        });
    }

    hookAll();
    colorizeMetrics(window.parent.document);
    // Poll frequently AND react to DOM mutations - polling alone left a
    // window (up to 500ms, longer under load) where a freshly-rerendered
    // panel sat uncolored; observing mutations recolors it the instant
    // Streamlit finishes swapping the DOM in, so it can no longer be
    // caught "between" polls.
    setInterval(function() { hookAll(); reclaimFocus(); colorizeMetrics(window.parent.document); }, 200);
    try {
        var mo = new MutationObserver(function() { colorizeMetrics(window.parent.document); });
        mo.observe(window.parent.document.body, { childList: true, subtree: true });
    } catch (e) {}
})();
</script>
""", height=0, width=0)

# --------------------------------------------------------------------------- #
# Video processor
# --------------------------------------------------------------------------- #

class FilterProcessor(VideoProcessorBase):
    def __init__(self):
        self.filter_name = "Original"
        self.mirror = True
        self.intensity = 100

        self.combo_enabled = False
        self.combo_filter = "Original"
        self.combo_intensity = 50

        self.save_requested = False
        self.last_saved_path = None

        self.strip_requested = False
        self._strip_active = False
        self._strip_frames = []
        self._strip_next_time = 0.0

        self.record_start_requested = False
        self.record_stop_requested = False
        self.is_recording = False
        self._writer = None
        self._record_path = None

    def _render(self, img):
        fn = FILTER_FUNCS.get(self.filter_name, FILTER_FUNCS["Original"])
        try:
            filtered = fn(img)
        except Exception:
            filtered = img

        alpha = max(0.0, min(1.0, self.intensity / 100.0))
        if self.filter_name != "Original" and alpha < 1.0:
            filtered = cv2.addWeighted(img, 1 - alpha, filtered, alpha, 0)

        if self.combo_enabled and self.combo_filter != "Original":
            combo_fn = FILTER_FUNCS.get(self.combo_filter, FILTER_FUNCS["Original"])
            try:
                combo_layer = combo_fn(filtered)
                c_alpha = max(0.0, min(1.0, self.combo_intensity / 100.0))
                filtered = cv2.addWeighted(filtered, 1 - c_alpha, combo_layer, c_alpha, 0)
            except Exception:
                pass

        return filtered

    def _make_strip(self):
        divider_color = _hex_to_bgr(FILTER_COLORS.get(self.filter_name, "#8b5cf6"))
        bar = 10
        frames = self._strip_frames
        h, w = frames[0].shape[:2]
        strip = np.full((h * 3 + bar * 4, w, 3), divider_color, dtype=np.uint8)
        y = bar
        for f in frames:
            strip[y:y + h, :, :] = f
            y += h + bar
        return strip

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)

        filtered = self._render(img)
        now = time.time()

        if self.save_requested:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{self.filter_name.replace(' ', '_').lower()}_{timestamp}.png"
            path = os.path.join(CAPTURES_DIR, fname)
            cv2.imwrite(path, filtered)
            self.last_saved_path = path
            self.save_requested = False

        if self.strip_requested and not self._strip_active:
            self._strip_active = True
            self._strip_frames = []
            self._strip_next_time = now
            self.strip_requested = False

        if self._strip_active and now >= self._strip_next_time:
            self._strip_frames.append(filtered.copy())
            self._strip_next_time = now + 0.7
            if len(self._strip_frames) >= 3:
                strip_img = self._make_strip()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"strip_{self.filter_name.replace(' ', '_').lower()}_{timestamp}.png"
                path = os.path.join(CAPTURES_DIR, fname)
                cv2.imwrite(path, strip_img)
                self.last_saved_path = path
                self._strip_active = False
                self._strip_frames = []

        if self.record_start_requested and not self.is_recording:
            h, w = filtered.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"clip_{self.filter_name.replace(' ', '_').lower()}_{timestamp}.mp4"
            path = os.path.join(CAPTURES_DIR, fname)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            self._record_path = path
            self.is_recording = True
            self.record_start_requested = False

        if self.is_recording and self._writer is not None:
            self._writer.write(filtered)

        if self.record_stop_requested and self.is_recording:
            if self._writer is not None:
                self._writer.release()
            self.last_saved_path = self._record_path
            self.is_recording = False
            self._writer = None
            self._record_path = None
            self.record_stop_requested = False

        return VideoFrame.from_ndarray(filtered, format="bgr24")


# --------------------------------------------------------------------------- #
# Nav bar
# --------------------------------------------------------------------------- #

n_shots_top = len(glob.glob(os.path.join(CAPTURES_DIR, "*.png"))) + len(glob.glob(os.path.join(CAPTURES_DIR, "*.mp4")))
if st.session_state.recording:
    status_html = '<span class="led led-rec"></span> Recording'
else:
    status_html = f'<span class="led led-on"></span> {n_shots_top} saved'

with st.container(key="nav_bar"):
    nav_brand, nav_status, nav_toggle = st.columns([5, 2, 1])
    with nav_brand:
        st.markdown(
            '<div class="nav-brand"><span class="dot"></span>\u2728 LUMINA STUDIO</div>',
            unsafe_allow_html=True,
        )
    with nav_status:
        st.markdown(f'<div class="status-pill">{status_html}</div>', unsafe_allow_html=True)
    with nav_toggle:
        if st.button("\u2600\ufe0f" if is_dark else "\U0001F319", key="theme_toggle_btn",
                     help="Switch to light mode" if is_dark else "Switch to dark mode"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

st.markdown(f"""
<div class="hero">
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.2rem;">Real-time <span class="accent-word">filters</span> for your webcam!</h1>
        <p style="color:var(--muted); font-size:0.98rem; margin:0.5rem 0 0 0; max-width:640px;">Ten filters you can stack into your own combos, a 3-2-1 photobooth countdown, photo strips, and video capture. All running locally, all controllable from the keyboard.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Main dashboard: video + controls
# --------------------------------------------------------------------------- #

col_video, col_controls = st.columns([2.1, 1], gap="large")

with col_video:
    st.markdown('<div class="marker-camera"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="panel_camera"):
        st.markdown('<div class="panel-title tint-camera">\U0001F3A5 Live Camera</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-hint">Click Start below, approve the camera prompt once, then control everything from the keyboard.</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="filters",
            video_processor_factory=FilterProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        ic1, ic2 = st.columns(2)
        with ic1:
            st.session_state.mirror = st.checkbox("Mirror preview", value=st.session_state.mirror)
        with ic2:
            st.session_state.intensity = st.slider("Effect intensity", 0, 100, st.session_state.intensity)
        if st.button("\u2328\ufe0f Re-enable keyboard shortcuts", key="reclaim_focus_btn", use_container_width=True):
            st.toast("Keyboard control re-enabled. Try the number keys now.", icon="\u2328\ufe0f")
        st.markdown('<div class="panel-hint" style="margin:0.5rem 0 0 0;">If number keys stop responding after clicking Start, click the button above once. Any normal click on this page (not on the video itself) returns keyboard focus here.</div>', unsafe_allow_html=True)

    st.markdown('<div class="marker-combo"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="panel_combo"):
        st.markdown('<div class="panel-title tint-combo">\U0001F9EA Combo Lab</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-hint">Layer a second filter on top of your active one for a look neither has alone.</div>', unsafe_allow_html=True)
        combo_c1, combo_c2, combo_c3 = st.columns([1, 1.4, 1])
        with combo_c1:
            st.session_state.combo_enabled = st.checkbox("Enable combo", value=st.session_state.combo_enabled)
        with combo_c2:
            st.session_state.combo_filter = st.selectbox(
                "Second filter", [f for f in FILTER_ORDER if f != "Original"],
                index=[f for f in FILTER_ORDER if f != "Original"].index(st.session_state.combo_filter)
                if st.session_state.combo_filter in FILTER_ORDER else 0,
                disabled=not st.session_state.combo_enabled,
            )
        with combo_c3:
            st.session_state.combo_intensity = st.slider(
                "Blend", 0, 100, st.session_state.combo_intensity, disabled=not st.session_state.combo_enabled
            )

with col_controls:
    combo_note = ""
    if st.session_state.combo_enabled and st.session_state.combo_filter != "Original":
        combo_note = f'<div class="combo-note">+ {FILTER_ICONS[st.session_state.combo_filter]} {st.session_state.combo_filter} @ {st.session_state.combo_intensity}%</div>'
    st.markdown(f"""
    <div class="active-filter-card">
        <div class="eyebrow">Active filter</div>
        <div class="value">{FILTER_ICONS[st.session_state.filter_name]}&nbsp; {st.session_state.filter_name}</div>
        {combo_note}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="marker-filters"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="panel_filters"):
        st.markdown('<div class="panel-title tint-filters">\U0001F3A8 Filters</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-hint">Press a number key anywhere on the page.</div>', unsafe_allow_html=True)

        grid = st.columns(2)
        for i, name in enumerate(FILTER_ORDER, start=1):
            key_label = str(i) if i <= 9 else "0"
            is_active = name == st.session_state.filter_name
            with grid[(i - 1) % 2]:
                if st.button(f"[{key_label}] {FILTER_ICONS[name]} {name}", key=f"filter_btn_{i}",
                             type="primary" if is_active else "secondary", use_container_width=True):
                    st.session_state.filter_name = name
                    st.rerun()

        if st.button("\U0001F3B2 Surprise me", key="random_btn", use_container_width=True):
            choices = [n for n in FILTER_ORDER if n != st.session_state.filter_name]
            st.session_state.filter_name = random.choice(choices)
            st.rerun()

    st.markdown('<div class="marker-capture"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="panel_capture"):
        st.markdown('<div class="panel-title tint-capture">\U0001F4F8 Capture</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-hint">Press <span class="kbd">s</span> to save, <span class="kbd">c</span> for a countdown, <span class="kbd">p</span> for a photo strip, <span class="kbd">r</span> to record.</div>', unsafe_allow_html=True)

        cap_c1, cap_c2 = st.columns(2)
        with cap_c1:
            if st.button("[S] Save frame", key="save_btn", type="primary", use_container_width=True):
                st.session_state.pending_save = True
        with cap_c2:
            st.button("\u23F3 Countdown", key="countdown_btn", use_container_width=True)

        if st.button("[P] \U0001F5BC\ufe0f Capture photo strip (3 shots)", key="strip_btn", use_container_width=True):
            st.session_state.pending_strip = True

        rec_label = "[R] \u23F9\ufe0f Stop recording" if st.session_state.recording else "[R] \U0001F534 Start recording"
        if st.button(rec_label, key="record_btn", use_container_width=True,
                     type="primary" if st.session_state.recording else "secondary"):
            st.session_state.recording = not st.session_state.recording
            st.rerun()

        kb_html = " ".join(f'<span class="kbd">{i}</span>' for i in range(1, 10)) + ' <span class="kbd">0</span>'
        st.markdown(f'<div class="panel-hint" style="margin:0.6rem 0 0 0;">{kb_html} switch filter</div>', unsafe_allow_html=True)

if ctx.video_processor:
    ctx.video_processor.filter_name = st.session_state.filter_name
    ctx.video_processor.mirror = st.session_state.mirror
    ctx.video_processor.intensity = st.session_state.intensity
    ctx.video_processor.combo_enabled = st.session_state.combo_enabled
    ctx.video_processor.combo_filter = st.session_state.combo_filter
    ctx.video_processor.combo_intensity = st.session_state.combo_intensity

    if st.session_state.pending_save:
        ctx.video_processor.save_requested = True
        st.session_state.pending_save = False
        st.rerun()

    if st.session_state.pending_strip:
        ctx.video_processor.strip_requested = True
        st.session_state.pending_strip = False
        st.toast("Capturing a 3-shot photo strip, hold still \u2728", icon="\U0001F5BC\ufe0f")

    if st.session_state.recording and not ctx.video_processor.is_recording:
        ctx.video_processor.record_start_requested = True
    if not st.session_state.recording and ctx.video_processor.is_recording:
        ctx.video_processor.record_stop_requested = True

    if (ctx.video_processor.last_saved_path
            and ctx.video_processor.last_saved_path != st.session_state.last_shown_capture):
        st.session_state.last_shown_capture = ctx.video_processor.last_saved_path
        st.toast(f"Saved {os.path.basename(ctx.video_processor.last_saved_path)}", icon="\u2705")

# --------------------------------------------------------------------------- #
# Stat row
# --------------------------------------------------------------------------- #

elapsed = datetime.now() - st.session_state.session_start
mins, secs = divmod(int(elapsed.total_seconds()), 60)
n_shots = len(glob.glob(os.path.join(CAPTURES_DIR, "*.png")))
n_clips = len(glob.glob(os.path.join(CAPTURES_DIR, "*.mp4")))
camera_live = bool(ctx.state.playing) if ctx else False

st.markdown('<div style="margin-top:0.4rem;"></div>', unsafe_allow_html=True)
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Camera status", "Live" if camera_live else "Off")
s2.metric("Active filter", st.session_state.filter_name)
s3.metric("Photos saved", n_shots)
s4.metric("Clips saved", n_clips)
s5.metric("Session length", f"{mins:02d}:{secs:02d}")

# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #

st.divider()

with st.container(border=True, key="panel_gallery"):
    st.markdown('<span class="tint-gallery" style="display:none"></span>', unsafe_allow_html=True)
    gcol1, gcol2, gcol3 = st.columns([3, 1, 1])
    with gcol1:
        st.subheader("Your Saved Shots")

    image_paths = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*.png")), reverse=True)
    video_paths = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*.mp4")), reverse=True)
    all_paths = sorted(image_paths + video_paths, key=os.path.getmtime, reverse=True)

    with gcol2:
        if all_paths:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for p in all_paths:
                    zf.write(p, arcname=os.path.basename(p))
            st.download_button("Download all", buf.getvalue(),
                                file_name="lumina_snapshots.zip", use_container_width=True)
    with gcol3:
        if st.button("Clear all", use_container_width=True):
            for p in all_paths:
                os.remove(p)
            st.session_state.last_shown_capture = None
            st.rerun()

    if not all_paths:
        st.markdown(
            '<div class="empty-state">No snapshots yet. Press a number key to pick a filter, '
            'then press <span class="kbd">s</span> to save, <span class="kbd">c</span> for a countdown, '
            'or <span class="kbd">p</span> for a photo strip.</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(4)
        for i, path in enumerate(all_paths):
            fname = os.path.basename(path)
            is_video = fname.endswith(".mp4")
            is_strip = fname.startswith("strip_")
            kind_class = "kind-video" if is_video else ("kind-strip" if is_strip else "kind-photo")
            badge_class = "badge-video" if is_video else ("badge-strip" if is_strip else "badge-photo")
            badge_text = "Clip" if is_video else ("Strip" if is_strip else "Photo")
            with cols[i % 4]:
                st.markdown(f'<div class="gallery-card {kind_class}">', unsafe_allow_html=True)
                st.markdown(f'<span class="gallery-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
                if is_video:
                    st.video(path)
                else:
                    st.image(path, use_container_width=True)
                st.markdown(f'<div class="gallery-caption">{fname}</div>', unsafe_allow_html=True)
                with open(path, "rb") as f:
                    st.download_button("Download", f, file_name=fname,
                                        key=f"dl_{path}", use_container_width=True)
                if st.button("Delete", key=f"del_{path}", use_container_width=True):
                    os.remove(path)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="site-footer">Lumina Studio. Built with Streamlit, OpenCV and WebRTC. '
    'All snapshots and clips stay on your own machine.</div>',
    unsafe_allow_html=True,
)