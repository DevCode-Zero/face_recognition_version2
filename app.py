"""
app.py
Flask web UI for face enrollment with Supabase + pgvector.
Requires SUPABASE_URL and SUPABASE_KEY environment variables.
"""

import os
import sys
import tempfile
import shutil

# Load env before other imports
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_engine import FaceEngine
from gallery import FaceGallery
from augment import augment_face_crop
from config import AUGMENT_COUNT, SIMILARITY_THRESHOLD, SUPABASE_URL, SUPABASE_KEY

import cv2
import numpy as np

if not SUPABASE_URL or not SUPABASE_KEY:
    print("=" * 60)
    print("WARNING: No Supabase credentials - using LOCAL storage")
    print("=" * 60)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

engine  = FaceEngine()
gallery = FaceGallery()

# ── HTML template ─────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Face Enrollment</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --teal:    #0D9488;
    --teal-l:  #CCFAF4;
    --teal-d:  #0a7a70;
    --slate:   #1E293B;
    --slate-m: #475569;
    --slate-l: #94A3B8;
    --bg:      #F7F9FC;
    --bg-card: #FFFFFF;
    --border:  #E2E8F0;
    --red:     #E11D48;
    --red-l:   #FFE4E6;
    --green:   #059669;
    --green-l: #D1FAE5;
    --amber:   #D97706;
    --amber-l: #FEF3C7;
    --radius:  12px;
    --shadow:  0 2px 12px rgba(0,0,0,0.07);
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--slate);
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    height: 60px;
  }
  .logo-dot {
    width: 10px; height: 10px;
    background: var(--teal);
    border-radius: 50%;
  }
  header h1 { font-size: 1rem; font-weight: 600; color: var(--slate); }
  header span { font-size: 0.82rem; color: var(--slate-l); margin-left: auto; font-family: 'DM Mono', monospace; }

  /* ── Layout ── */
  .container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 1.5rem;
    align-items: start;
  }

  /* ── Card ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .card-header {
    padding: 1.1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .card-header h2 { font-size: 0.95rem; font-weight: 600; }
  .badge {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 99px;
    background: var(--teal-l);
    color: var(--teal);
    font-family: 'DM Mono', monospace;
  }
  .card-body { padding: 1.4rem; }

  /* ── Form ── */
  label.field-label {
    display: block;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--slate-m);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  input[type="text"] {
    width: 100%;
    padding: 0.7rem 0.9rem;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--slate);
    background: var(--bg);
    outline: none;
    transition: border-color 0.15s;
    margin-bottom: 1.2rem;
  }
  input[type="text"]:focus { border-color: var(--teal); background: #fff; }

  /* ── Drop zone ── */
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 2.5rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
    margin-bottom: 1.2rem;
  }
  .dropzone:hover, .dropzone.drag-over {
    border-color: var(--teal);
    background: var(--teal-l);
  }
  .dropzone input[type="file"] {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    opacity: 0; cursor: pointer;
  }
  .drop-icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .drop-text { font-size: 0.9rem; color: var(--slate-m); }
  .drop-sub  { font-size: 0.78rem; color: var(--slate-l); margin-top: 0.25rem; }

  /* ── Preview strip ── */
  .previews {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    min-height: 0;
  }
  .preview-item {
    position: relative;
    width: 80px; height: 80px;
    border-radius: 8px;
    overflow: hidden;
    border: 1.5px solid var(--border);
    animation: pop-in 0.2s ease;
  }
  @keyframes pop-in { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  .preview-item img { width: 100%; height: 100%; object-fit: cover; }
  .preview-item .remove-btn {
    position: absolute; top: 3px; right: 3px;
    width: 18px; height: 18px;
    background: rgba(0,0,0,0.55);
    border: none; border-radius: 50%;
    color: #fff; font-size: 11px;
    cursor: pointer; display: flex;
    align-items: center; justify-content: center;
    line-height: 1;
  }

  /* ── Button ── */
  .btn-enroll {
    width: 100%;
    padding: 0.85rem;
    background: var(--teal);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }
  .btn-enroll:hover:not(:disabled) { background: var(--teal-d); }
  .btn-enroll:active:not(:disabled) { transform: scale(0.98); }
  .btn-enroll:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Toast ── */
  #toast {
    position: fixed;
    bottom: 1.5rem; right: 1.5rem;
    padding: 0.75rem 1.2rem;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transform: translateY(100px);
    opacity: 0;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    z-index: 999;
    max-width: 320px;
  }
  #toast.show { transform: translateY(0); opacity: 1; }
  #toast.success { background: var(--green); color: #fff; }
  #toast.error   { background: var(--red); color: #fff; }

  /* ── Progress ── */
  .progress-wrap {
    display: none;
    margin-bottom: 1rem;
    background: var(--border);
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    background: var(--teal);
    border-radius: 99px;
    width: 0%;
    transition: width 0.3s ease;
    animation: shimmer 1.5s infinite;
  }
  @keyframes shimmer {
    0%   { opacity: 1; }
    50%  { opacity: 0.7; }
    100% { opacity: 1; }
  }

  /* ── Gallery list ── */
  .gallery-list { display: flex; flex-direction: column; gap: 0.5rem; }
  .person-row {
    display: flex;
    align-items: center;
    padding: 0.65rem 0.9rem;
    border-radius: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    gap: 0.75rem;
    animation: pop-in 0.2s ease;
  }
  .person-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: var(--teal-l);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 600; color: var(--teal);
    flex-shrink: 0;
    text-transform: uppercase;
  }
  .person-info { flex: 1; min-width: 0; }
  .person-name { font-weight: 600; font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .person-count { font-size: 0.75rem; color: var(--slate-l); font-family: 'DM Mono', monospace; }
  .btn-remove {
    padding: 0.3rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: transparent;
    color: var(--slate-l);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .btn-remove:hover { border-color: var(--red); color: var(--red); background: var(--red-l); }

  .empty-state {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--slate-l);
    font-size: 0.88rem;
  }
  .empty-state .empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }

  /* ── Stats bar ── */
  .stats-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border-bottom: 1px solid var(--border);
  }
  .stat-item {
    padding: 0.9rem 1rem;
    text-align: center;
    border-right: 1px solid var(--border);
  }
  .stat-item:last-child { border-right: none; }
  .stat-val { font-size: 1.4rem; font-weight: 600; color: var(--teal); font-family: 'DM Mono', monospace; }
  .stat-lbl { font-size: 0.72rem; color: var(--slate-l); margin-top: 2px; }

  /* ── Nav tabs ── */
  .nav-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }
  .nav-tab {
    padding: 0.75rem 1.5rem;
    border: none;
    background: none;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--slate-l);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }
  .nav-tab:hover { color: var(--slate); }
  .nav-tab.active {
    color: var(--teal);
    border-bottom-color: var(--teal);
  }
  .tab-content { display: none; }
  .tab-content.active { display: contents; }

  /* ── Recognition styles ── */
  .recognize-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 2rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .recognize-zone:hover, .recognize-zone.drag-over {
    border-color: var(--teal);
    background: var(--teal-l);
  }
  .recognize-zone input[type="file"] {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    opacity: 0; cursor: pointer;
  }
  .recognize-preview {
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    display: none;
  }
  .recognize-preview.visible { display: block; }
  .result-overlay {
    position: relative;
    display: inline-block;
  }
  .result-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    border-radius: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    margin-bottom: 0.5rem;
  }
  .result-item.known { border-left: 3px solid var(--green); }
  .result-item.unknown { border-left: 3px solid var(--red); }
  .result-name { font-weight: 600; }
  .result-score { font-size: 0.8rem; color: var(--slate-l); font-family: 'DM Mono', monospace; }
  .confidence-bar {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.25rem;
  }
  .confidence-fill {
    height: 100%;
    background: var(--green);
    border-radius: 3px;
    transition: width 0.3s ease;
  }
  .confidence-fill.low { background: var(--amber); }
  .confidence-fill.very-low { background: var(--red); }

  @media (max-width: 768px) {
    .container { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>Face Recognition</h1>
  <span id="header-status">Ready</span>
</header>

<!-- Nav tabs -->
<div class="nav-tabs">
  <button class="nav-tab active" onclick="switchTab('enroll')">Enroll</button>
  <button class="nav-tab" onclick="switchTab('recognize')">Recognize</button>
</div>

<div class="container">

  <!-- Tab: Enroll -->
  <div id="tab-enroll" class="tab-content active">
    <!-- LEFT: Enroll form -->
    <div>
      <div class="card">
        <div class="card-header">
          <h2>Enroll New Person</h2>
          <span class="badge">ArcFace + 20× augment</span>
        </div>
        <div class="card-body">

          <label class="field-label">Full name</label>
          <input type="text" id="name-input" placeholder="e.g. Shivam Kumar" autocomplete="off"/>

          <label class="field-label">Photos (1–5 images)</label>
          <div class="dropzone" id="dropzone">
            <input type="file" id="file-input" accept="image/*" multiple/>
            <div class="drop-icon">📷</div>
            <div class="drop-text">Drop photos here or click to browse</div>
            <div class="drop-sub">JPG, PNG, WEBP · Max 10MB each</div>
          </div>

          <div class="previews" id="previews"></div>

          <div class="progress-wrap" id="progress-wrap">
            <div class="progress-bar" id="progress-bar"></div>
          </div>

          <button class="btn-enroll" id="enroll-btn" disabled onclick="enrollPerson()">
            <span id="btn-icon">✦</span>
            <span id="btn-text">Enroll Person</span>
          </button>

        </div>
      </div>
    </div>

    <!-- RIGHT: Gallery -->
    <div>
      <div class="card">
        <div class="stats-bar">
          <div class="stat-item">
            <div class="stat-val" id="stat-people">0</div>
            <div class="stat-lbl">People</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" id="stat-embeddings">0</div>
            <div class="stat-lbl">Embeddings</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" id="stat-ready">–</div>
            <div class="stat-lbl">Status</div>
          </div>
        </div>
        <div class="card-header">
          <h2>Enrolled People</h2>
        </div>
        <div class="card-body" style="padding: 1rem;">
          <div class="gallery-list" id="gallery-list">
            <div class="empty-state">
              <div class="empty-icon">👤</div>
              Loading gallery…
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab: Recognize -->
  <div id="tab-recognize" class="tab-content">
    <!-- LEFT: Recognition zone -->
    <div>
      <div class="card">
        <div class="card-header">
          <h2>Recognize Faces</h2>
          <span class="badge">ArcFace + FAISS</span>
        </div>
        <div class="card-body">
          <div class="recognize-zone" id="recognize-zone">
            <input type="file" id="recognize-input" accept="image/*"/>
            <div class="drop-icon">🔍</div>
            <div class="drop-text">Drop an image here or click to browse</div>
            <div class="drop-sub">JPG, PNG, WEBP · Max 10MB</div>
          </div>
          
          <!-- Webcam section -->
          <div id="webcam-section" style="margin-top: 1rem; display: none;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
              <span style="font-size: 0.8rem; font-weight: 500; color: var(--slate-m); text-transform: uppercase;">Or use webcam</span>
            </div>
            <video id="webcam-video" autoplay playsinline style="width: 100%; max-height: 300px; border-radius: 8px; display: block;"></video>
            <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">
              <button id="btn-capture" onclick="capturePhoto()" style="flex: 1; padding: 0.75rem; background: var(--teal); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-family: 'DM Sans', sans-serif;">
                📸 Capture
              </button>
              <button id="btn-stop-webcam" onclick="stopWebcam()" style="padding: 0.75rem 1rem; background: var(--bg); color: var(--slate-m); border: 1px solid var(--border); border-radius: 8px; cursor: pointer;">
                ✕ Stop
              </button>
            </div>
          </div>
          
          <button id="btn-start-webcam" onclick="startWebcam()" style="width: 100%; margin-top: 1rem; padding: 0.75rem; background: var(--bg); color: var(--slate-m); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; font-family: 'DM Sans', sans-serif;">
            📹 Use Webcam
          </button>
          
          <div id="recognize-result" style="margin-top: 1rem;"></div>
        </div>
      </div>
    </div>

    <!-- RIGHT: Results -->
    <div>
      <div class="card">
        <div class="card-header">
          <h2>Recognition Results</h2>
        </div>
        <div class="card-body" id="result-container">
          <div class="empty-state">
            <div class="empty-icon">📸</div>
            Upload an image to see results
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
const nameInput  = document.getElementById('name-input');
const fileInput  = document.getElementById('file-input');
const dropzone   = document.getElementById('dropzone');
const previews   = document.getElementById('previews');
const enrollBtn  = document.getElementById('enroll-btn');
const btnText    = document.getElementById('btn-text');
const btnIcon    = document.getElementById('btn-icon');
const progressW  = document.getElementById('progress-wrap');
const progressB  = document.getElementById('progress-bar');

let selectedFiles = [];

// ── Drag & drop ───────────────────────────────────────────────
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => {
  addFiles([...fileInput.files]);
  fileInput.value = '';
});
nameInput.addEventListener('input', checkReady);

function addFiles(files) {
  files.filter(f => f.type.startsWith('image/')).forEach(f => {
    if (selectedFiles.length >= 5) return;
    if (selectedFiles.find(x => x.name === f.name && x.size === f.size)) return;
    selectedFiles.push(f);
    addPreview(f);
  });
  checkReady();
}

function addPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const div = document.createElement('div');
    div.className = 'preview-item';
    div.dataset.name = file.name;
    div.innerHTML = `
      <img src="${e.target.result}" alt="${file.name}"/>
      <button class="remove-btn" onclick="removeFile('${file.name}')">✕</button>`;
    previews.appendChild(div);
  };
  reader.readAsDataURL(file);
}

function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  const el = previews.querySelector(`[data-name="${name}"]`);
  if (el) el.remove();
  checkReady();
}

function checkReady() {
  enrollBtn.disabled = !(nameInput.value.trim() && selectedFiles.length > 0);
}

// ── Enroll ────────────────────────────────────────────────────
async function enrollPerson() {
  const name = nameInput.value.trim();
  if (!name || selectedFiles.length === 0) return;

  enrollBtn.disabled = true;
  btnText.textContent = 'Enrolling…';
  btnIcon.textContent = '⟳';
  progressW.style.display = 'block';
  progressB.style.width = '30%';

  const formData = new FormData();
  formData.append('name', name);
  selectedFiles.forEach(f => formData.append('images', f));

  try {
    progressB.style.width = '60%';
    const res = await fetch('/enroll', { method: 'POST', body: formData });
    const data = await res.json();
    progressB.style.width = '100%';

    if (data.success) {
      showToast(`✓ ${name} enrolled with ${data.embeddings} embeddings`, 'success');
      resetForm();
      loadGallery();
    } else {
      showToast(`✗ ${data.error}`, 'error');
    }
  } catch (err) {
    showToast('✗ Network error. Is the server running?', 'error');
  } finally {
    setTimeout(() => {
      progressW.style.display = 'none';
      progressB.style.width = '0%';
      btnText.textContent = 'Enroll Person';
      btnIcon.textContent = '✦';
      checkReady();
    }, 600);
  }
}

function resetForm() {
  nameInput.value = '';
  selectedFiles = [];
  previews.innerHTML = '';
  enrollBtn.disabled = true;
}

// ── Remove ────────────────────────────────────────────────────
async function removePerson(name) {
  if (!confirm(`Remove "${name}" from gallery?`)) return;
  const res = await fetch('/remove', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  const data = await res.json();
  if (data.success) {
    showToast(`Removed ${name}`, 'success');
    loadGallery();
  } else {
    showToast(data.error, 'error');
  }
}

// ── Gallery ───────────────────────────────────────────────────
async function loadGallery() {
  const res  = await fetch('/gallery');
  const data = await res.json();

  document.getElementById('stat-people').textContent     = data.people;
  document.getElementById('stat-embeddings').textContent = data.total_embeddings;
  document.getElementById('stat-ready').textContent      = data.people > 0 ? 'Ready' : '–';
  document.getElementById('header-status').textContent   = data.people > 0 ? `${data.people} enrolled` : 'Ready';

  const list = document.getElementById('gallery-list');

  if (data.entries.length === 0) {
    list.innerHTML = `<div class="empty-state"><div class="empty-icon">👤</div>No people enrolled yet.<br/>Add someone above to get started.</div>`;
    return;
  }

  list.innerHTML = data.entries.map(e => `
    <div class="person-row">
      <div class="person-avatar">${e.name.charAt(0)}</div>
      <div class="person-info">
        <div class="person-name">${e.name}</div>
        <div class="person-count">${e.count} embeddings</div>
      </div>
      <button class="btn-remove" onclick="removePerson('${e.name}')">Remove</button>
    </div>`).join('');
}

// ── Toast ─────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = '', 3500);
}

// ── Init ──────────────────────────────────────────────────────
loadGallery();
setInterval(loadGallery, 10000);

// ── Tab switching ───────────────────────────────────────────────
function switchTab(tab) {
  const targetUrl = tab === 'recognize' ? '/recognize' : '/enroll';
  history.pushState(null, '', targetUrl);
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`.nav-tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
}

// Initialize tab from URL
function initTabFromUrl() {
  const path = window.location.pathname;
  if (path === '/recognize') {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.nav-tab[onclick="switchTab('recognize')"]`).classList.add('active');
    document.getElementById('tab-recognize').classList.add('active');
  } else {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.nav-tab[onclick="switchTab('enroll')"]`).classList.add('active');
    document.getElementById('tab-enroll').classList.add('active');
  }
}
document.addEventListener('DOMContentLoaded', initTabFromUrl);

// ── Recognition ─────────────────────────────────────────────────
const recognizeZone = document.getElementById('recognize-zone');
const recognizeInput = document.getElementById('recognize-input');
const recognizeResult = document.getElementById('recognize-result');
const resultContainer = document.getElementById('result-container');

recognizeZone.addEventListener('dragover', e => { e.preventDefault(); recognizeZone.classList.add('drag-over'); });
recognizeZone.addEventListener('dragleave', () => recognizeZone.classList.remove('drag-over'));
recognizeZone.addEventListener('drop', e => {
  e.preventDefault();
  recognizeZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    processRecognizeImage(e.dataTransfer.files[0]);
  }
});
recognizeInput.addEventListener('change', () => {
  if (recognizeInput.files.length > 0) {
    processRecognizeImage(recognizeInput.files[0]);
  }
});

async function processRecognizeImage(file) {
  if (!file.type.startsWith('image/')) return;

  recognizeResult.innerHTML = '<div style="text-align:center;padding:2rem;">⟳ Processing...</div>';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('/api/recognize', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      recognizeResult.innerHTML = `<div style="color:var(--red);padding:1rem;">✗ ${data.error}</div>`;
      return;
    }

    // Show image with bounding boxes
    const imgUrl = URL.createObjectURL(file);
    let boxesHtml = '';
    data.faces.forEach(f => {
      const color = f.name === 'unknown' ? 'var(--red)' : 'var(--green)';
      boxesHtml += `
        <div class="result-item ${f.name === 'unknown' ? 'unknown' : 'known'}">
          <div class="result-name" style="color:${color}">${f.name}</div>
          <div class="result-score">Confidence: ${(f.score * 100).toFixed(1)}%</div>
          <div class="confidence-bar">
            <div class="confidence-fill ${f.score < 0.5 ? 'very-low' : f.score < 0.7 ? 'low' : ''}" style="width:${f.score * 100}%"></div>
          </div>
        </div>
      `;
    });

    // Show preview with boxes (percentage-based for responsive sizing)
    let boxesOverlay = '';
    data.faces.forEach(f => {
      const color = f.name === 'unknown' ? '255,0,0' : '0,255,0';
      const x1pct = (f.x1 / data.img_width * 100).toFixed(2);
      const y1pct = (f.y1 / data.img_height * 100).toFixed(2);
      const w = ((f.x2 - f.x1) / data.img_width * 100).toFixed(2);
      const h = ((f.y2 - f.y1) / data.img_height * 100).toFixed(2);
      boxesOverlay += `<div style="position:absolute;left:${x1pct}%;top:${y1pct}%;width:${w}%;height:${h}%;border:3px solid rgb(${color});border-radius:4px;box-sizing:border-box;"></div>`;
      boxesOverlay += `<div style="position:absolute;left:${x1pct}%;top:calc(${y1pct}% - 22px);background:rgb(${color});color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;white-space:nowrap;">${f.name} (${(f.score*100).toFixed(0)}%)</div>`;
    });

    recognizeResult.innerHTML = `
      <div class="result-overlay" style="position:relative;display:inline-block;">
        <img src="${imgUrl}" class="recognize-preview visible" style="max-width:100%;border-radius:8px;"/>
        <div style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;">${boxesOverlay}</div>
      </div>
    `;

    // Update result list
    if (data.faces.length === 0) {
      resultContainer.innerHTML = `<div class="empty-state"><div class="empty-icon">😕</div>No faces detected</div>`;
    } else {
      resultContainer.innerHTML = `
        <div style="margin-bottom:1rem;font-weight:600;color:var(--slate-m);">${data.faces.length} face(s) detected</div>
        ${boxesHtml}
      `;
    }

    showToast(`Found ${data.faces.length} face(s)`, 'success');
  } catch (err) {
    recognizeResult.innerHTML = `<div style="color:var(--red);padding:1rem;">✗ Network error</div>`;
  }
}

// ── Webcam functions ─────────────────────────────────────────────
let webcamStream = null;
const webcamVideo = document.getElementById('webcam-video');
const webcamSection = document.getElementById('webcam-section');
const btnStartWebcam = document.getElementById('btn-start-webcam');

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamVideo.srcObject = webcamStream;
    webcamSection.style.display = 'block';
    btnStartWebcam.style.display = 'none';
    recognizeZone.style.display = 'none';
  } catch (err) {
    showToast('Camera access denied. Please allow camera permission.', 'error');
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  webcamVideo.srcObject = null;
  webcamSection.style.display = 'none';
  btnStartWebcam.style.display = 'block';
  recognizeZone.style.display = 'flex';
}

function capturePhoto() {
  const canvas = document.createElement('canvas');
  canvas.width = webcamVideo.videoWidth;
  canvas.height = webcamVideo.videoHeight;
  canvas.getContext('2d').drawImage(webcamVideo, 0, 0);
  
  canvas.toBlob(blob => {
    const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
    processRecognizeImage(file);
  }, 'image/jpeg', 0.95);
}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html><head>
<title>Face Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
body { font-family: 'DM Sans', sans-serif; background: #f0f4f8; margin: 0; padding: 2rem; }
.container { max-width: 600px; margin: 0 auto; }
.card { background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden; margin-bottom: 1.5rem; }
.card-header { background: #0D9488; color: white; padding: 1.5rem; text-align: center; }
.card-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; }
.card-body { padding: 2rem; text-align: center; }
.btn { display: inline-block; padding: 1rem 2rem; background: #0D9488; color: white; text-decoration: none; border-radius: 8px; font-size: 1.1rem; font-weight: 600; margin: 0.5rem; }
.btn:hover { background: #0a7a70; }
.btn-secondary { background: #64748b; }
.btn-secondary:hover { background: #475569; }
p { color: #64748b; }
</style>
</head><body>
<div class="container">
<div class="card">
<div class="card-header"><h1>Face Recognition</h1></div>
<div class="card-body">
<p>Welcome to Face Recognition System</p>
<a href="/enroll" class="btn">Enroll</a>
<a href="/recognize" class="btn btn-secondary">Recognize</a>
</div>
</div>
</div>
</body></html>
"""


@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    if request.method == "GET":
        tab_html = HTML.replace('class="nav-tab active" onclick="switchTab(\'recognize\')"', 'class="nav-tab" onclick="switchTab(\'recognize\')"')
        tab_html = tab_html.replace('class="tab-content active"', 'class="tab-content"').replace('id="tab-recognize" class="tab-content"', 'id="tab-recognize" class="tab-content active"')
        return render_template_string(tab_html)


@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "GET":
        tab_html = HTML.replace('class="nav-tab active" onclick="switchTab(\'enroll\')"', 'class="nav-tab" onclick="switchTab(\'enroll\')"')
        tab_html = tab_html.replace('class="tab-content active"', 'class="tab-content"').replace('id="tab-enroll" class="tab-content"', 'id="tab-enroll" class="tab-content active"')
        return render_template_string(tab_html)
    
    name   = request.form.get("name", "").strip()
    images = request.files.getlist("images")

    if not name:
        return jsonify({"success": False, "error": "Name is required"})
    if not images:
        return jsonify({"success": False, "error": "At least one image is required"})

    all_embeddings = []
    tmpdir = tempfile.mkdtemp()

    try:
        for img_file in images:
            tmp_path = os.path.join(tmpdir, img_file.filename)
            img_file.save(tmp_path)

            img = cv2.imread(tmp_path)
            if img is None:
                continue

            face = engine.get_largest_face(img)
            if face is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            pad = int((x2 - x1) * 0.15)
            h, w = img.shape[:2]
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            crop = img[y1:y2, x1:x2]

            variants = augment_face_crop(crop, n=AUGMENT_COUNT)
            for v in variants:
                emb = engine.embed_crop(v)
                if emb is not None:
                    all_embeddings.append(emb)

            if face.embedding is not None:
                all_embeddings.append(face.embedding)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not all_embeddings:
        return jsonify({"success": False, "error": "No face detected in any uploaded image. Please use a clear, front-facing photo."})

    gallery.add_embeddings(name, all_embeddings)
    gallery.save()

    return jsonify({"success": True, "name": name, "embeddings": len(all_embeddings)})


@app.route("/remove", methods=["POST"])
def remove():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name required"})
    gallery.remove_person(name)
    gallery.save()
    return jsonify({"success": True})


@app.route("/gallery")
def get_gallery():
    counts  = gallery.embedding_count_per_person()
    entries = [{"name": n, "count": c} for n, c in sorted(counts.items())]
    return jsonify({
        "people":           len(entries),
        "total_embeddings": sum(c["count"] for c in entries),
        "entries":          entries,
    })


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """JSON API for enrolling a person with base64 image data"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "JSON data required"})
    
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"})
    
    import base64
    img_data = data.get("image")
    if not img_data:
        return jsonify({"success": False, "error": "Image data required"})
    
    try:
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image data: {e}"})
    
    all_embeddings = []
    face = engine.get_largest_face(img)
    if face is None:
        return jsonify({"success": False, "error": "No face detected"})
    
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad = int((x2 - x1) * 0.15)
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    
    variants = augment_face_crop(crop, n=AUGMENT_COUNT)
    for v in variants:
        emb = engine.embed_crop(v)
        if emb is not None:
            all_embeddings.append(emb)
    
    if face.embedding is not None:
        all_embeddings.append(face.embedding)
    
    if not all_embeddings:
        return jsonify({"success": False, "error": "Failed to generate embeddings"})
    
    gallery.add_embeddings(name, all_embeddings)
    gallery.save()
    
    return jsonify({"success": True, "name": name, "embeddings": len(all_embeddings)})


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image provided", "faces": [], "img_width": 0, "img_height": 0})

    img_file = request.files["image"]
    if not img_file:
        return jsonify({"error": "No image file", "faces": [], "img_width": 0, "img_height": 0})

    tmp_path = os.path.join(tempfile.gettempdir(), img_file.filename)
    img_file.save(tmp_path)

    img = cv2.imread(tmp_path)
    if img is None:
        return jsonify({"error": "Could not read image", "faces": [], "img_width": 0, "img_height": 0})

    img_height, img_width = img.shape[:2]
    faces = engine.get_faces(img)
    results = []

    for face in faces:
        if face.embedding is None:
            continue
        name, score = gallery.search(face.embedding, threshold=SIMILARITY_THRESHOLD)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        results.append({
            "name": name,
            "score": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    try:
        os.remove(tmp_path)
    except:
        pass

    return jsonify({"faces": results, "img_width": img_width, "img_height": img_height})


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import PORT
    print("\n  Face Enrollment Web UI")
    print("  ===================")
    print(f"  Open: http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)