"""
KIỂM TRA TOÀN DIỆN BỘ DỮ LIỆU DEFACTIFY
==========================================
Kiểm tra:
  1. Metadata: số dòng, số cột, split counts, group integrity
  2. File ảnh trên ổ đĩa: đủ số lượng, không thiếu, không thừa
  3. Data Leakage: caption trùng giữa các tập
  4. Hash ảnh: các nhóm bị leakage có thực sự là cùng 1 ảnh không?
"""
import os
import sys
import hashlib
import pandas as pd
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = r"D:\nam_4\DATN\dataset\Defactify_dataset"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
VALID_COLS = ['group_id', 'filepath', 'split', 'label_a', 'label_b', 'model_name', 'caption']

EXPECTED = {
    "train":      {"groups": 7000,  "total": 42000, "real": 7000,  "ai": 35000},
    "validation": {"groups": 1500,  "total": 9000,  "real": 1500,  "ai": 7500},
    "test":       {"groups": 7500,  "total": 45000, "real": 7500,  "ai": 37500},
}

PASS = "[OK]"
FAIL = "[!!]"
WARN = "[??]"

def sep(char="=", n=70): print(char * n)
def ok(msg):   print(f"  {PASS}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}")
def warn(msg): print(f"  {WARN}  {msg}")

# ─────────────────────────────────────────────
#  Đọc metadata an toàn
# ─────────────────────────────────────────────
def load_df():
    for enc in ['utf-8', 'utf-8-sig', 'windows-1252', 'latin1']:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc, usecols=VALID_COLS)
            print(f"  -> Đọc CSV thành công với encoding: [{enc}]")
            return df
        except UnicodeDecodeError:
            continue
        except ValueError:
            try:
                df = pd.read_csv(CSV_PATH, encoding=enc)
                df = df[[c for c in VALID_COLS if c in df.columns]]
                print(f"  -> Đọc CSV thành công (fallback) với encoding: [{enc}]")
                return df
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError("Không thể đọc metadata.csv với bất kỳ encoding phổ biến nào!")

# ─────────────────────────────────────────────
#  MD5 hash của file ảnh
# ─────────────────────────────────────────────
def file_hash(filepath, chunk=1024*1024):
    h = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            data = f.read(chunk)
            h.update(data)
        return h.hexdigest()
    except Exception:
        return None

# ─────────────────────────────────────────────
#  PHẦN 1: Kiểm tra Metadata
# ─────────────────────────────────────────────
def check_metadata(df):
    sep()
    print("PHAN 1: KIEM TRA METADATA.CSV")
    sep()

    n = len(df)
    print(f"\n[1.1] Tong so dong trong metadata: {n:,}")
    if n == 96000:
        ok("Dung 96.000 dong.")
    else:
        fail(f"Sai! Can 96.000, dang co {n:,}")

    dup = df['filepath'].duplicated().sum()
    print(f"\n[1.2] Filepath trung lap: {dup}")
    if dup == 0:
        ok("Khong co filepath trung lap.")
    else:
        fail(f"Co {dup} filepath bi trung! Anh se bi ghi de.")

    print("\n[1.3] Kiem tra tung tap du lieu:")
    all_ok = True
    for split, exp in EXPECTED.items():
        df_s = df[df['split'] == split]
        total = len(df_s)
        groups = df_s['group_id'].nunique()
        real_rows = (df_s['label_a'] == 0).sum()
        ai_rows   = (df_s['label_a'] == 1).sum()

        print(f"\n  >>> Tap: {split.upper()}")
        marker = PASS if total == exp['total'] else FAIL
        print(f"  [{marker}] Tong: {total:,} / {exp['total']:,}")
        if total != exp['total']: all_ok = False

        marker = PASS if groups == exp['groups'] else FAIL
        print(f"  [{marker}] Groups: {groups:,} / {exp['groups']:,}")
        if groups != exp['groups']: all_ok = False

        marker = PASS if (real_rows == exp['real'] and ai_rows == exp['ai']) else FAIL
        print(f"  [{marker}] Real: {real_rows:,} (can {exp['real']:,}) | AI: {ai_rows:,} (can {exp['ai']:,})")
        if real_rows != exp['real'] or ai_rows != exp['ai']: all_ok = False

        bad_count = df_s.groupby('group_id').size()
        bad_groups_cnt = (bad_count != 6).sum()
        if bad_groups_cnt == 0:
            ok(f"Moi group deu co dung 6 anh (1 Real + 5 AI).")
        else:
            fail(f"{bad_groups_cnt} group khong co dung 6 anh!")
            all_ok = False

    return all_ok

# ─────────────────────────────────────────────
#  PHẦN 2: Kiểm tra File Ảnh trên Ổ Đĩa
# ─────────────────────────────────────────────
def check_disk_files(df):
    sep()
    print("PHAN 2: KIEM TRA FILE ANH TREN O DIA")
    sep()

    missing = []
    zero_size = []
    total = len(df)

    print(f"\n  Dang quet {total:,} file anh... (co the mat 1-2 phut)")
    checked = 0
    for _, row in df.iterrows():
        abs_path = os.path.join(BASE_DIR, row['filepath'])
        if not os.path.exists(abs_path):
            missing.append(row['filepath'])
        elif os.path.getsize(abs_path) == 0:
            zero_size.append(row['filepath'])
        checked += 1
        if checked % 10000 == 0:
            print(f"  ... da kiem tra {checked:,}/{total:,}")

    print(f"\n[2.1] File anh BI THIEU (trong metadata nhung khong co tren o dia):")
    if len(missing) == 0:
        ok(f"Khong co file nao bi thieu! Tat ca {total:,} file deu ton tai.")
    else:
        fail(f"THIEU {len(missing):,} file!")
        for p in missing[:10]:
            print(f"         - {p}")
        if len(missing) > 10:
            print(f"         ... va {len(missing) - 10} file nua.")

    print(f"\n[2.2] File anh bi RONG (0 bytes):")
    if len(zero_size) == 0:
        ok("Khong co file nao rong.")
    else:
        fail(f"CO {len(zero_size):,} file rong!")
        for p in zero_size[:5]:
            print(f"         - {p}")

    print("\n[2.3] So file thuc te tren o dia theo tung tap:")
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(BASE_DIR, split)
        real_dir = os.path.join(split_dir, "real")
        ai_dir   = os.path.join(split_dir, "ai")
        n_real = len(list(Path(real_dir).glob("*.jpg"))) if os.path.exists(real_dir) else 0
        n_ai   = len(list(Path(ai_dir).glob("*.jpg")))   if os.path.exists(ai_dir)   else 0
        exp = EXPECTED[split]
        mr = PASS if n_real == exp['real'] else FAIL
        ma = PASS if n_ai   == exp['ai']   else FAIL
        print(f"  {split.upper():12} | Real: [{mr}] {n_real:>5}/{exp['real']:>5}  |  AI: [{ma}] {n_ai:>6}/{exp['ai']:>6}")

    return len(missing), len(zero_size)

# ─────────────────────────────────────────────
#  PHẦN 3: Kiểm tra Data Leakage
# ─────────────────────────────────────────────
def check_data_leakage(df):
    sep()
    print("PHAN 3: KIEM TRA DATA LEAKAGE")
    sep()

    train_caps = set(df[df['split'] == 'train']['caption'].dropna().unique())
    val_caps   = set(df[df['split'] == 'validation']['caption'].dropna().unique())
    test_caps  = set(df[df['split'] == 'test']['caption'].dropna().unique())

    tv   = train_caps & val_caps
    tt   = train_caps & test_caps
    vt   = val_caps   & test_caps
    all3 = train_caps & val_caps & test_caps

    print(f"\n  Caption duy nhat | TRAIN: {len(train_caps):,} | VAL: {len(val_caps):,} | TEST: {len(test_caps):,}")
    print(f"\n  Leakage:")
    print(f"    TRAIN x VAL  : {len(tv):>4} captions")
    print(f"    TRAIN x TEST : {len(tt):>4} captions")
    print(f"    VAL   x TEST : {len(vt):>4} captions")
    print(f"    Ca 3 tap     : {len(all3):>4} captions")

    if len(tt) > 0 or len(tv) > 0:
        fail("CO DATA LEAKAGE! Captions xuat hien o nhieu tap.")
    else:
        ok("Khong co Data Leakage.")

    return tt, tv, vt

# ─────────────────────────────────────────────
#  PHẦN 4: Xác nhận bằng Hash Ảnh
# ─────────────────────────────────────────────
def confirm_leakage_with_hash(df, train_test_overlap, max_samples=3):
    sep()
    print("PHAN 4: XAC NHAN LEAKAGE BANG HASH FILE ANH")
    sep("-")
    print("  (So sanh MD5 cua anh Real trong Train vs Real trong Test cung caption)")
    sep("-")

    if not train_test_overlap:
        ok("Khong co caption trung de kiem tra.")
        return

    confirmed_same = 0
    confirmed_diff = 0

    for idx, cap in enumerate(list(train_test_overlap)[:max_samples]):
        train_real = df[(df['split'] == 'train') & (df['caption'] == cap) & (df['label_a'] == 0)]
        test_real  = df[(df['split'] == 'test')  & (df['caption'] == cap) & (df['label_a'] == 0)]

        if train_real.empty or test_real.empty:
            continue

        short_cap = cap[:60] + "..." if len(cap) > 60 else cap
        print(f"\n  [{idx+1}] Caption: \"{short_cap}\"")

        train_hashes = {}
        for _, row in train_real.iterrows():
            p = os.path.join(BASE_DIR, row['filepath'])
            h = file_hash(p)
            exists = os.path.exists(p)
            train_hashes[row['group_id']] = h
            print(f"       TRAIN group {row['group_id']:>5} | File: {'OK' if exists else 'MISS':4} | MD5: {(h or 'N/A')[:12]}...")

        for _, row in test_real.iterrows():
            p = os.path.join(BASE_DIR, row['filepath'])
            h = file_hash(p)
            exists = os.path.exists(p)
            print(f"       TEST  group {row['group_id']:>5} | File: {'OK' if exists else 'MISS':4} | MD5: {(h or 'N/A')[:12]}...")

            for train_gid, train_h in train_hashes.items():
                if train_h and h:
                    if h == train_h:
                        print(f"         *** HASH KHOP HOAN TOAN: test_group {row['group_id']} GIONG HET train_group {train_gid}!")
                        confirmed_same += 1
                    else:
                        confirmed_diff += 1

    sep("-")
    print(f"\n  Ket qua so sanh hash:")
    if confirmed_same > 0:
        fail(f"{confirmed_same} cap anh GIONG NHAU HOAN TOAN (hash khop 100%) - Duplicate nghiem trong!")
    if confirmed_diff > 0:
        warn(f"{confirmed_diff} cap cung caption nhung hash KHAC nhau - Augmented variant (loi nguy co trung binh).")
    if confirmed_same == 0 and confirmed_diff == 0:
        ok("Khong tim thay anh nao de so sanh.")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sep("=")
    print("  KIEM TRA TOAN DIEN BO DU LIEU DEFACTIFY")
    sep("=")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  CSV path : {CSV_PATH}")
    sep("=")

    if not os.path.exists(CSV_PATH):
        fail(f"Khong tim thay metadata.csv tai: {CSV_PATH}")
        sys.exit(1)

    print("\nDang doc metadata.csv ...")
    df = load_df()
    print(f"  -> Tong so dong: {len(df):,} | Cot: {list(df.columns)}")

    meta_ok              = check_metadata(df)
    n_missing, n_zero    = check_disk_files(df)
    tt_overlap, tv_overlap, vt_overlap = check_data_leakage(df)
    confirm_leakage_with_hash(df, tt_overlap, max_samples=3)

    sep()
    print("TONG KET")
    sep()
    if meta_ok:
        ok("Metadata: DAT - So dong, groups, ty le Real/AI deu dung.")
    else:
        fail("Metadata: KHONG DAT - Co sai sot trong metadata!")

    if n_missing == 0 and n_zero == 0:
        ok("File anh: DAT - Tat ca anh deu ton tai va khong rong.")
    else:
        fail(f"File anh: KHONG DAT - {n_missing} anh thieu, {n_zero} anh rong!")

    if len(tt_overlap) == 0 and len(tv_overlap) == 0:
        ok("Data Leakage: KHONG CO - Cac tap hoan toan doc lap.")
    else:
        fail(f"Data Leakage: CO - {len(tt_overlap)} caption trung Train<->Test, {len(tv_overlap)} caption trung Train<->Val.")
    sep()
