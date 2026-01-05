#!/usr/bin/env python
"""
Upload trained classification models to Hugging Face

Usage:
1. Set HF_TOKEN environment variable or enter token when prompted
2. Run: python upload_models_to_huggingface.py

Structure created on HF:
v1_model/
├── BTCUSDT/
│   ├── 15m/
│   │   ├── classification.h5
│   │   └── params.json
│   └── 1h/
│       ├── classification.h5
│       └── params.json
├── ETHUSDT/
│   ├── 15m/
│   │   ├── classification.h5
│   │   └── params.json
└── ... (all 22 symbols)
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

def get_hf_token():
    """Get Hugging Face token from environment or user input"""
    token = os.getenv('HF_TOKEN')
    if not token:
        token = input('Enter your Hugging Face token: ')
    return token

def upload_models_to_huggingface(
    models_dir,
    hf_repo_id='zongowo111/v2-crypto-ohlcv-data',
    hf_token=None
):
    """
    Upload models from local directory to Hugging Face
    
    Args:
        models_dir: Path to models directory (e.g., /content/drive/MyDrive/crypto-zigzag-ml/models)
        hf_repo_id: Hugging Face repo ID (dataset)
        hf_token: Hugging Face token
    """
    
    if not hf_token:
        hf_token = get_hf_token()
    
    api = HfApi(token=hf_token)
    
    print('='*70)
    print('UPLOADING MODELS TO HUGGING FACE')
    print('='*70)
    print(f'Models directory: {models_dir}')
    print(f'HF Repo: {hf_repo_id}')
    print(f'Target structure: v1_model/<SYMBOL>/<TIMEFRAME>/')
    print()
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f'Error: Models directory not found: {models_dir}')
        return False
    
    # Get all model directories
    model_dirs = sorted([d for d in models_path.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print('No model directories found')
        return False
    
    print(f'Found {len(model_dirs)} model directories')
    print('\nStarting upload...\n')
    
    total_uploaded = 0
    total_failed = 0
    
    for model_dir in model_dirs:
        # Parse symbol and timeframe from directory name
        # Format: btcusdt_15m or ethusdt_1h
        dir_name = model_dir.name
        parts = dir_name.rsplit('_', 1)
        
        if len(parts) != 2:
            print(f'⊘ Skipping {dir_name}: Invalid format')
            continue
        
        symbol_lower, timeframe = parts
        symbol = symbol_lower.upper()
        
        print(f'Processing {symbol} {timeframe}...', end='', flush=True)
        
        # Get files to upload
        h5_file = model_dir / 'classification.h5'
        params_file = model_dir / 'params.json'
        
        if not h5_file.exists() or not params_file.exists():
            print(f' ⊘ Missing files')
            total_failed += 1
            continue
        
        try:
            # Read files
            with open(h5_file, 'rb') as f:
                h5_content = f.read()
            
            with open(params_file, 'r') as f:
                params_content = f.read()
            
            # Create path structure: v1_model/SYMBOL/TIMEFRAME/
            h5_hf_path = f'v1_model/{symbol}/{timeframe}/classification.h5'
            params_hf_path = f'v1_model/{symbol}/{timeframe}/params.json'
            
            # Prepare upload operations
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
            
            print(f' ✓')
            total_uploaded += 1
            
        except Exception as e:
            print(f' ✗ {str(e)[:50]}')
            total_failed += 1
    
    print()
    print('='*70)
    print('UPLOAD SUMMARY')
    print('='*70)
    print(f'Successfully uploaded: {total_uploaded}/{len(model_dirs)}')
    print(f'Failed: {total_failed}')
    print()
    print('Structure created on Hugging Face:')
    print(f'https://huggingface.co/datasets/{hf_repo_id}/tree/main/v1_model')
    print()
    print('='*70)
    
    return total_failed == 0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload models to Hugging Face')
    parser.add_argument(
        '--models-dir',
        default='/content/drive/MyDrive/crypto-zigzag-ml/models',
        help='Path to models directory'
    )
    parser.add_argument(
        '--hf-repo',
        default='zongowo111/v2-crypto-ohlcv-data',
        help='Hugging Face repo ID'
    )
    parser.add_argument(
        '--hf-token',
        default=None,
        help='Hugging Face token (or set HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    success = upload_models_to_huggingface(
        args.models_dir,
        args.hf_repo,
        args.hf_token
    )
    
    exit(0 if success else 1)
