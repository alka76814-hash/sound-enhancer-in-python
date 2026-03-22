"""
build_dsp.py — Compile the C++ DSP extension
=============================================
Run this once from your project folder:

    python build_dsp.py

Output:
    Windows  → dsp_core.pyd   (loadable by ctypes as dsp_core.dll)
    macOS    → dsp_core.so
    Linux    → dsp_core.so

Requirements:
    Windows: MinGW-w64 (g++) OR MSVC (cl.exe)
             Install MinGW: https://winlibs.com  (just unzip, add bin/ to PATH)
    macOS:   Xcode command line tools  (xcode-select --install)
    Linux:   g++  (sudo apt install g++)
"""

import subprocess
import sys
import os
import shutil

SRC   = "dsp_core.cpp"
FLAGS = ["-O3", "-march=native", "-ffast-math", "-std=c++17"]

def detect_compiler():
    for cc in ["g++", "c++", "clang++"]:
        if shutil.which(cc):
            return cc
    # Windows MSVC
    if shutil.which("cl"):
        return "msvc"
    return None

def build():
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found. Make sure dsp_core.cpp is in this folder.")
        sys.exit(1)

    cc = detect_compiler()
    if cc is None:
        print("ERROR: No C++ compiler found.")
        print("  Windows: install MinGW-w64 from https://winlibs.com")
        print("  macOS:   xcode-select --install")
        print("  Linux:   sudo apt install g++")
        sys.exit(1)

    if sys.platform == "win32":
        out = "dsp_core.pyd"
    else:
        out = "dsp_core.so"

    if cc == "msvc":
        cmd = ["cl", "/O2", "/LD", "/EHsc", SRC, f"/Fe:{out}"]
    else:
        cmd = [cc, *FLAGS, "-shared", "-fPIC", "-o", out, SRC]

    print(f"Compiling with {cc}...")
    print("  " + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILATION FAILED:")
        print(result.stderr)
        sys.exit(1)

    size_kb = os.path.getsize(out) / 1024
    print(f"\n✓ Built {out}  ({size_kb:.0f} KB)")
    print(f"  Place it in the same folder as engine.py and main.py.")
    print(f"  The app will auto-detect and use it — numpy is the fallback.")

if __name__ == "__main__":
    build()