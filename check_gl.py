#!/usr/bin/env python3
"""check_gl.py — diagnose EGL/OpenGL before starting a long run.

    python check_gl.py

Runs the import chain that fails in benchmark.py, one link at a time, so you
see which one breaks instead of a traceback ten frames deep.
"""
import os, sys, glob

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
print(f"MUJOCO_GL          = {os.environ['MUJOCO_GL']}")
print(f"PYOPENGL_PLATFORM  = {os.environ['PYOPENGL_PLATFORM']}")
print(f"DISPLAY            = {os.environ.get('DISPLAY', '(unset)')}")

# EGL needs a registered vendor driver; an empty directory is the usual cause
# of PLATFORM.EGL being None on a fresh cloud image.
icd = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json"))
print(f"EGL vendor ICDs    = {[os.path.basename(i) for i in icd] or 'NONE — this is the problem'}")

try:
    import OpenGL.platform as _p
    print(f"PyOpenGL platform  = {type(_p.PLATFORM).__name__}")
    print(f"  .EGL attribute   = {'present' if getattr(_p.PLATFORM, 'EGL', None) else 'None  <- broken'}")
except Exception as e:
    sys.exit(f"PyOpenGL import failed: {e}")

try:
    from OpenGL import EGL
    print("from OpenGL import EGL   OK")
except Exception as e:
    print(f"from OpenGL import EGL   FAILED: {type(e).__name__}: {e}")
    print("\nFix, in order:")
    print("  export PYOPENGL_PLATFORM=egl")
    print("  sudo apt-get install -y libegl1 libegl-mesa0 libgl1 libglvnd0 libopengl0")
    print("  if the ICD list above is empty, reinstall the NVIDIA driver with --install-libglvnd")
    print("  or fall back to CPU rendering:  export MUJOCO_GL=osmesa")
    sys.exit(1)

import mujoco
print(f"mujoco {mujoco.__version__} imported")
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><light pos='0 0 1'/>"
    "<geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64)
r.update_scene(mujoco.MjData(m))
img = r.render()
r.close()
print(f"offscreen render OK — {img.shape}, mean pixel {img.mean():.1f}")
print("\nGL is working. Safe to run benchmark.py and train_3dof.py.")