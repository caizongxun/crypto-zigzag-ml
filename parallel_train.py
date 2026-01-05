#!/usr/bin/env python
"""
並行訓練管理器 - 同時跑多個 worker
使用方式: python parallel_train.py
"""
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from itertools import product

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'MATICUSDT', 'LINKUSDT', 'LITUSDT', 'UNIUSDT',
    'AVAXUSDT', 'SOLUUSDT', 'FTMUSDT', 'AAVEUSDT', 'CRVUSDT',
    'MKRUSDT', 'SNXUSDT', 'COMPUSDT', 'LRCUSDT', 'GRTUSDT',
    'ALGOUSDT', 'ATOMUSDT'
]

TIMEFRAMES = ['15m', '1h']
MAX_WORKERS = 3  # RTX 4090: 建議用 3 或 4

class ParallelTrainer:
    def __init__(self):
        self.running_processes = {}
        self.completed = []
        self.failed = []
        self.queue = list(product(SYMBOLS, TIMEFRAMES))
        self.start_time = None
    
    def start_job(self, symbol, timeframe, gpu_id):
        """啟動訓練任務"""
        cmd = f'python train_worker.py {gpu_id} {symbol} {timeframe}'
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        job_id = f'{symbol}_{timeframe}'
        self.running_processes[job_id] = {
            'process': proc,
            'symbol': symbol,
            'timeframe': timeframe,
            'gpu_id': gpu_id,
            'start_time': datetime.now()
        }
        print(f'[QUEUE] Started {symbol} {timeframe} on GPU {gpu_id}')
    
    def check_completed(self):
        """検查已完成的任務"""
        completed_jobs = []
        for job_id, job_info in self.running_processes.items():
            proc = job_info['process']
            if proc.poll() is not None:  # 進程已完成
                stdout, stderr = proc.communicate()
                elapsed = (datetime.now() - job_info['start_time']).total_seconds() / 60
                
                symbol = job_info['symbol']
                timeframe = job_info['timeframe']
                
                if proc.returncode == 0:
                    print(f'[DONE] {symbol} {timeframe} ({elapsed:.1f}m)')
                    self.completed.append(job_id)
                else:
                    print(f'[FAIL] {symbol} {timeframe} ({elapsed:.1f}m)')
                    if stderr:
                        error_msg = stderr.decode()[:100]
                        print(f'       Error: {error_msg}')
                    self.failed.append(job_id)
                
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            del self.running_processes[job_id]
    
    def run(self):
        """主訓練迴圈"""
        self.start_time = datetime.now()
        print(f'\n{"="*70}')
        print(f'Parallel Training - RTX 4090')
        print(f'{"="*70}')
        print(f'Total jobs: {len(self.queue)}')
        print(f'Max workers: {MAX_WORKERS}')
        avg_time_per_job = 0.3  # hours
        estimated_hours = len(self.queue) / MAX_WORKERS * avg_time_per_job
        print(f'Estimated time: {estimated_hours:.1f} hours ({estimated_hours*60:.0f} minutes)')
        print(f'\nStarting training...\n')
        
        while self.queue or self.running_processes:
            # 検查已完成的任務
            self.check_completed()
            
            # 提交新任務
            while len(self.running_processes) < MAX_WORKERS and self.queue:
                symbol, timeframe = self.queue.pop(0)
                gpu_id = len(self.running_processes) % MAX_WORKERS
                self.start_job(symbol, timeframe, gpu_id)
                time.sleep(2)  # 避免同時啟動過多進程
            
            # 等待
            if self.running_processes:
                time.sleep(15)  # 例需穷比検查
            
            # 進度報告
            total = len(self.completed) + len(self.failed) + len(self.queue)
            completed = len(self.completed) + len(self.failed)
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            
            if completed > 0:
                avg_time = elapsed / completed
                remaining_mins = avg_time * len(self.queue)
                remaining_hours = remaining_mins / 60
                print(f'[PROGRESS] {completed}/{total} | OK: {len(self.completed)} | Failed: {len(self.failed)} | Running: {len(self.running_processes)} | Time left: {remaining_hours:.1f}h')
        
        # 最終報告
        total_time_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        print(f'\n{"="*70}')
        print(f'BATCH TRAINING COMPLETED')
        print(f'{"="*70}')
        print(f'Total time: {total_time_hours:.1f} hours')
        print(f'Completed: {len(self.completed)}/{len(self.completed) + len(self.failed)}')
        print(f'Failed: {len(self.failed)}')
        
        if self.failed:
            print(f'\nFailed jobs:')
            for job in self.failed:
                print(f'  - {job}')
        
        print(f'\n{"="*70}')
        print(f'Models saved to: ./models/')
        print(f'{"="*70}\n')

if __name__ == '__main__':
    # 確保有 models 目錄
    Path('models').mkdir(exist_ok=True)
    
    trainer = ParallelTrainer()
    try:
        trainer.run()
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user')
        for job_info in trainer.running_processes.values():
            try:
                job_info['process'].terminate()
            except:
                pass
