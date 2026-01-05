#!/usr/bin/env python
"""
Colab Script - Upload Models to Hugging Face

Usage in Colab:
1. !pip install huggingface-hub -q
2. exec(open('colab_upload_to_huggingface.py').read())

Structure created:
v1_model/
├── BTCUSDT/
│   ├── 15m/
│   │   ├── classification.h5
│   │   └── params.json
│   └── 1h/
│       ├── classification.h5
│       └── params.json
├── ETHUSDT/
│   └── ...
└── ... (all 22 symbols)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, CommitOperationAdd
except ImportError:
    print('Installing huggingface-hub...')
    os.system('pip install huggingface-hub -q')
    from huggingface_hub import HfApi, CommitOperationAdd

def upload_models_to_huggingface(
    models_dir='/content/drive/MyDrive/crypto-zigzag-ml/models',
    hf_repo_id='zongowo111/v2-crypto-ohlcv-data',
    hf_token=None
):
    """
    Upload classification models to Hugging Face
    """
    
    print('='*70)
    print('COLAB MODEL UPLOADER - HUGGING FACE')
    print('='*70)
    print()
    
    # Get token
    if not hf_token:
        hf_token = input('Enter your Hugging Face token: ')
    
    if not hf_token.strip():
        print('Error: No token provided')
        return False
    
    # Initialize API
    try:
        api = HfApi(token=hf_token)
        print('Hugging Face API initialized')
    except Exception as e:
        print(f'Error: Invalid token - {e}')
        return False
    
    print(f'Repo: {hf_repo_id}')
    print(f'Models dir: {models_dir}')
    print()
    
    # Check models directory
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f'Error: Models directory not found: {models_dir}')
        return False
    
    # Get all model directories
    model_dirs = sorted([d for d in models_path.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print('Error: No model directories found')
        return False
    
    print(f'Found {len(model_dirs)} model directories')
    print()
    print('-'*70)
    print('Starting upload...')
    print('-'*70)
    print()
    
    total_uploaded = 0
    total_failed = 0
    failed_models = []
    uploaded_models = []
    
    start_time = datetime.now()
    
    for idx, model_dir in enumerate(model_dirs, 1):
        # Parse directory name
        dir_name = model_dir.name
        parts = dir_name.rsplit('_', 1)
        
        if len(parts) != 2:
            print(f'[{idx:2d}/{len(model_dirs)}] {dir_name:20s} SKIP (invalid format)')
            continue
        
        symbol_lower, timeframe = parts
        symbol = symbol_lower.upper()
        
        print(f'[{idx:2d}/{len(model_dirs)}] {symbol:10s} {timeframe:3s}... ', end='', flush=True)
        
        # Check files
        h5_file = model_dir / 'classification.h5'
        params_file = model_dir / 'params.json'
        
        if not h5_file.exists() or not params_file.exists():
            print('FAIL (missing files)')
            total_failed += 1
            failed_models.append((symbol, timeframe, 'missing files'))
            continue
        
        try:
            # Prepare paths
            h5_hf_path = f'v1_model/{symbol}/{timeframe}/classification.h5'
            params_hf_path = f'v1_model/{symbol}/{timeframe}/params.json'
            
            # Get file sizes
            h5_size = h5_file.stat().st_size / (1024*1024)  # MB
            params_size = params_file.stat().st_size / 1024  # KB
            
            # Prepare operations
            operations = [
                CommitOperationAdd(
                    path_in_repo=h5_hf_path,
                    path_or_fileobj=h5_file
                ),
                CommitOperationAdd(
                    path_in_repo=params_hf_path,
                    path_or_fileobj=params_file
                )
            ]
            
            # Upload
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type='dataset',
                operations=operations,
                commit_message=f'Add {symbol} {timeframe} classification model'
            )
            
            print(f'OK ({h5_size:.1f}MB)')
            total_uploaded += 1
            uploaded_models.append((symbol, timeframe))
            
        except Exception as e:
            error_msg = str(e)[:50]
            print(f'FAIL ({error_msg})')
            total_failed += 1
            failed_models.append((symbol, timeframe, error_msg))
    
    elapsed_time = (datetime.now() - start_time).total_seconds() / 60
    
    print()
    print('='*70)
    print('UPLOAD SUMMARY')
    print('='*70)
    print(f'Total time: {elapsed_time:.1f} minutes')
    print(f'Successfully uploaded: {total_uploaded}/{len(model_dirs)}')
    print(f'Failed: {total_failed}')
    print()
    
    if uploaded_models:
        print('Uploaded models:')
        for symbol, timeframe in uploaded_models[:10]:
            print(f'  ✓ {symbol} {timeframe}')
        if len(uploaded_models) > 10:
            print(f'  ... and {len(uploaded_models)-10} more')
    
    if failed_models:
        print()
        print('Failed models:')
        for symbol, timeframe, error in failed_models[:10]:
            print(f'  ✗ {symbol} {timeframe}: {error}')
        if len(failed_models) > 10:
            print(f'  ... and {len(failed_models)-10} more')
    
    print()
    print('='*70)
    print('View your models:')
    print(f'https://huggingface.co/datasets/{hf_repo_id}/tree/main/v1_model')
    print('='*70)
    
    return total_failed == 0

if __name__ == '__main__':
    # Auto-detect if running in Colab
    try:
        from google.colab import drive
        # If we reach here, we're in Colab
        print('Running in Google Colab')
        print()
        
        # Mount Google Drive if not already mounted
        try:
            os.listdir('/content/drive/MyDrive')
            print('Google Drive already mounted')
        except:
            print('Mounting Google Drive...')
            drive.mount('/content/drive')
        
        print()
        
        # Run upload
        success = upload_models_to_huggingface()
        sys.exit(0 if success else 1)
        
    except ImportError:
        # Not in Colab, run as normal script
        print('Running as normal Python script')
        print()
        
        import argparse
        parser = argparse.ArgumentParser(description='Upload models to Hugging Face')
        parser.add_argument('--models-dir', required=True, help='Path to models directory')
        parser.add_argument('--hf-repo', default='zongowo111/v2-crypto-ohlcv-data', help='HF repo ID')
        parser.add_argument('--hf-token', help='HF token (or use HF_TOKEN env var)')
        
        args = parser.parse_args()
        hf_token = args.hf_token or os.getenv('HF_TOKEN')
        
        success = upload_models_to_huggingface(
            args.models_dir,
            args.hf_repo,
            hf_token
        )
        sys.exit(0 if success else 1)
