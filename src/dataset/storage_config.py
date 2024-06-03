import os


PROJECT_ROOT = os.path.join("..", "..")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_ROOT = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_ROOT = os.path.join(DATA_ROOT, "processed")
FULL_UMLS_ROOT = os.path.join(RAW_DATA_ROOT, "2024AA-full", "2024AA")
META_ROOT = os.path.join(FULL_UMLS_ROOT, "META")
MRCONSO_PATH = os.path.join(META_ROOT, "MRCONSO.RRF")
MRDEF_PATH = os.path.join(META_ROOT, "MRDEF.RRF")
VOCAB_PATH = os.path.join(META_ROOT, "VOCAB.csv")
