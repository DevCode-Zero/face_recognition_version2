import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY", "")
TABLE_PREFIX  = os.getenv("TABLE_PREFIX", "facerec_")

INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
DET_SIZE          = (640, 640)
CTX_ID            = -1

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
EMBEDDING_DIM        = 512
AUGMENT_COUNT        = int(os.getenv("AUGMENT_COUNT", "20"))

BOX_COLOR_KNOWN   = (0, 255, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)
FONT_SCALE        = 0.8
THICKNESS         = 2

PORT   = int(os.getenv("PORT", "5001"))
DEBUG  = os.getenv("DEBUG", "false").lower() == "true"
