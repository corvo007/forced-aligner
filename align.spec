# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata
import sys
import os

block_cipher = None

# =========================================================
# Configuration
# =========================================================
# CHANGE THIS: Path to your local ffmpeg.exe
FFMPEG_PATH = "ffmpeg.exe" 
# Checks if ffmpeg exists at build time
if not os.path.exists(FFMPEG_PATH):
    print(f"WARNING: '{FFMPEG_PATH}' not found. The built EXE will rely on external ffmpeg.")
    ts_binaries = []
else:
    print(f"INFO: Bundling '{FFMPEG_PATH}' into the executable.")
    # Bundle it to the root of the unpack directory (.)
    ts_binaries = [(FFMPEG_PATH, '.')]

# =========================================================
# Dependency Collection
# =========================================================
datas = []
binaries = []
hiddenimports = ['torch', 'ctc_forced_aligner', 'transformers']

# ctc_forced_aligner has data files (punctuations.lst)
tmp_ret = collect_all('ctc_forced_aligner')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# uroman has data files that must be included
tmp_ret = collect_all('uroman')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Metadata needed for runtime version checks (fixes importlib.metadata.PackageNotFoundError)
datas += copy_metadata('torchcodec')
datas += copy_metadata('transformers')
datas += copy_metadata('torch')

# Explicitly add wav2vec2 model imports
hiddenimports += [
    'transformers.models.wav2vec2',
    'transformers.models.wav2vec2.modeling_wav2vec2',
    'transformers.models.wav2vec2.configuration_wav2vec2',
    'transformers.models.wav2vec2.tokenization_wav2vec2',
    'transformers.models.wav2vec2.processing_wav2vec2',
]

# Add FFmpeg binaries
binaries += ts_binaries

a = Analysis(
    ['align.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='align',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='align',
)
