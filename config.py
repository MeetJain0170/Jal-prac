"""
config.py — Central configuration for JalDrishti Maritime Security Vision System.
All modules import from here. Change values once, effects propagate everywhere.
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
ENHANCED_DIR = os.path.join(DATA_DIR, 'enhanced')
SAMPLE_DIR   = os.path.join(DATA_DIR, 'sample')

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR     = os.path.join(BASE_DIR, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR    = os.path.join(OUTPUT_DIR, 'results')
LOGS_DIR       = os.path.join(OUTPUT_DIR, 'logs')

for _d in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Model architecture ───────────────────────────────────────────────────────
IMAGE_HEIGHT       = 256
IMAGE_WIDTH        = 256
IMAGE_CHANNELS     = 3
UNET_BASE_FEATURES = 64
UNET_INIT_FEATURES = UNET_BASE_FEATURES   # alias kept for train.py

# ── Patch inference ───────────────────────────────────────────────────────────
PATCH_SIZE    = 256
PATCH_OVERLAP = 64
BLEND_ALPHA   = 0.85   # weight for model output; 1.0=pure model, 0.0=pure original

# ── Training hyper-parameters ────────────────────────────────────────────────
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 80
TRAIN_SPLIT    = 0.8
RANDOM_SEED    = 42
GRAD_CLIP_NORM = 1.0

# ── Loss weights ──────────────────────────────────────────────────────────────
LOSS_W_L1      = 0.45
LOSS_W_SSIM    = 0.25
LOSS_W_PERC    = 0.15
LOSS_W_EDGE    = 0.15
LOSS_W_TV      = 0.15  # Total Variation to prevent static hallucination
LOSS_W_PHYSICS = 0.20  # Dark channel scattering to prevent magenta bleeds

LOSS_SCALE_FULL    = 1.0
LOSS_SCALE_HALF    = 0.4
LOSS_SCALE_QUARTER = 0.2

# ── LR Scheduler ─────────────────────────────────────────────────────────────
USE_LR_SCHEDULER = True
LR_MIN           = 1e-6

# ── Early stopping ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC   = "psnr"
SAVE_FREQ           = 10

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0
PIN_MEMORY  = torch.cuda.is_available()

# ── Evaluation & visualisation ───────────────────────────────────────────────
NUM_VISUALIZATION_SAMPLES = 5
SAVE_ALL_PREDICTIONS      = False

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_WEIGHTS     = 'yolov8s-world.pt'
YOLO_CONF_THRESH = 0.08
YOLO_IOU_THRESH  = 0.30
YOLO_IMG_SIZE    = 1280

# ── Depth estimation ─────────────────────────────────────────────────────────
MIDAS_MODEL_TYPE = "MiDaS_small"

# ── Water quality ────────────────────────────────────────────────────────────
TURBIDITY_DARK_PATCH_PCT = 0.001

# ── Video processing ─────────────────────────────────────────────────────────
VIDEO_MAX_FRAMES  = 300
VIDEO_FPS_DEFAULT = 25
VIDEO_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, 'videos')
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ── Data pipeline ────────────────────────────────────────────────────────────
DATASET_TYPE      = 'paired'
USE_AUGMENTATION  = False
AUGMENTATION_PROB = 0.5

# ── Logging ───────────────────────────────────────────────────────────────────
PRINT_FREQ = 10
VERBOSE    = True