import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_split_dataframes(
    metadata_path: str = r"D:\nam_4\DATN\dataset\Defactify_dataset\metadata.csv",
    known_models: Optional[List[str]] = None,
    unseen_models: Optional[List[str]] = None #list các model AI để huấn luyện và test
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    BƯỚC 1: Nạp và phân chia dữ liệu Known/Unseen trên bộ nhớ RAM.
    - metadata.csv trên đĩa giữ nguyên 100% (94,272 dòng).
    - train_df và val_df: Chỉ giữ ảnh Thật (MS_COCO) và các mô hình Known AI.
    - test_df: Giữ nguyên toàn bộ 6 mô hình (43,800 ảnh) để đánh giá cả Known và Unseen.
    """
    if known_models is None:
        known_models = ['MS_COCO', 'Stable_Diffusion_2.1', 'Stable_Diffusion_XL', 'DALL-E_3']
    if unseen_models is None:
        unseen_models = ['Stable_Diffusion_3', 'Midjourney_V6']
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata tại: {metadata_path}")
        
    df = pd.read_csv(metadata_path)
    print(f"-> Nạp thành công metadata : {len(df):,} dòng.")
    
    # 1. Phân chia tập TRAIN trên RAM
    train_df = df[(df['split'] == 'train') & (df['model_name'].isin(known_models))].copy() #lọc ra các dòng mô hình chứa các mô hình để huấn luyện
    train_df.reset_index(drop=True, inplace=True) # đánh số index lại tăng dần
    
    # 2. Phân chia tập VALIDATION trên RAM
    val_df = df[(df['split'] == 'validation') & (df['model_name'].isin(known_models))].copy()
    val_df.reset_index(drop=True, inplace=True)
    
    # 3. Phân chia tập TEST trên RAM (Giữ toàn bộ cả 6 model)
    test_df = df[df['split'] == 'test'].copy()
    test_df.reset_index(drop=True, inplace=True)
    
    print(f"-> Train DataFrame (Known only) : {len(train_df):,} ảnh (Real: {(train_df['label_a']==0).sum():,}, AI: {(train_df['label_a']==1).sum():,})")
    print(f"-> Val DataFrame   (Known only) : {len(val_df):,} ảnh (Real: {(val_df['label_a']==0).sum():,}, AI: {(val_df['label_a']==1).sum():,})")
    print(f"-> Test DataFrame  (Toàn bộ 6 model): {len(test_df):,} ảnh (Real: {(test_df['label_a']==0).sum():,}, AI: {(test_df['label_a']==1).sum():,})")
    print("=" * 70)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    get_split_dataframes()