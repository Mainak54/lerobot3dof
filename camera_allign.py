import os
import cv2
import numpy as np
import yaml
import mujoco

# Ensure offscreen rendering uses EGL so it doesn't conflict with OpenCV windows
os.environ.setdefault("MUJOCO_GL", "egl")

# --- Configuration ---
CONFIG_PATH = "configs/stage1_3dof.yaml"
CAMERA_INDEX = 2  # Change if your webcam is on /dev/video2, etc.
# ---------------------

def draw_alignment_guides(img):
    """Draws a crosshair and bounding box to help align the frame."""
    h, w = img.shape[:2]
    color = (0, 255, 0)
    # Center crosshairs
    cv2.line(img, (w // 2, 0), (w // 2, h), color, 1)
    cv2.line(img, (0, h // 2), (w, h // 2), color, 1)
    # Inner bounding box (useful for checking FOV scaling)
    cv2.rectangle(img, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), color, 1)
    return img

def main():
    # 1. Load config
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config {CONFIG_PATH}: {e}")
        return

    cam_name = cfg["env"].get("camera", "wrist_cam")
    w = int(cfg["detector"]["width"])
    h = int(cfg["detector"]["height"])
    scene_path = cfg["env"].get("scene", "mujoco/train_scene.xml")

    # 2. Setup MuJoCo Scene
    print(f"Loading MuJoCo scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Put the arm in the home position so it matches a known state
    home_qpos = cfg["env"]["home_qpos"]
    trained_joints = cfg["env"]["trained_joints"]
    for i, jnt_name in enumerate(trained_joints):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            q_adr = model.jnt_qposadr[jnt_id]
            data.qpos[q_adr] = home_qpos[i]
            
    mujoco.mj_forward(model, data)
    
    # Initialize the renderer
    renderer = mujoco.Renderer(model, height=h, width=w)

    # 3. Setup Hardware Camera
    print(f"Opening physical camera at index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print("Error: Could not open physical camera.")
        return

    print("\n--- Controls ---")
    print(" 'B' : Toggle Blend / Side-by-Side mode")
    print(" 'G' : Toggle Alignment Guides (Crosshairs)")
    print(" 'Q' or ESC : Quit")
    
    blend_mode = False
    show_guides = True

    try:
        while True:
            # Get Physical Frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
                
            # OpenCV captures BGR, MuJoCo uses RGB. Let's work in BGR for display.
            phys_bgr = frame.copy()

            # Get Sim Frame
            renderer.update_scene(data, camera=cam_name)
            sim_rgb = renderer.render()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # Draw guides
            if show_guides:
                phys_bgr = draw_alignment_guides(phys_bgr)
                sim_bgr = draw_alignment_guides(sim_bgr)

            # Display
            if blend_mode:
                # 50/50 Alpha blend overlaid on top of each other
                display_img = cv2.addWeighted(phys_bgr, 0.5, sim_bgr, 0.5, 0)
                cv2.putText(display_img, "BLENDED OVERLAY MODE", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Side by side
                display_img = np.hstack([phys_bgr, sim_bgr])
                cv2.putText(display_img, "PHYSICAL CAMERA", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, "SIMULATION CAMERA", (w + 10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Camera Alignment", display_img)

            # Key handlers
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('b'):
                blend_mode = not blend_mode
            elif key == ord('g'):
                show_guides = not show_guides

    finally:
        cap.release()
        renderer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()