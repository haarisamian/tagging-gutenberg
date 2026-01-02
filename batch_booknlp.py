import argparse
import re
import sys
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import torch
from booknlp.booknlp import BookNLP

def sanitize(title):
    s = re.sub(r'[<>:"/\\|?*]', '', title)
    s = re.sub(r'\s+', '_', s.strip())[:50]
    return s.strip('_') or 'unnamed'


def get_pgids(val):
    out = []
    for s in str(val).split(';'):
        s = s.strip().upper().replace('PG', '')
        if s.isdigit():
            out.append(int(s))
    return out

def process_book(job, device):
    pgid, title, in_path, out_dir = job
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    bnlp = BookNLP('en', {
        'pipeline': 'entity,quote,supersense,event,coref',
        'model': 'big',
        'device': device
    })
    bnlp.process(in_path, out_dir, f'PG{pgid}')
    return pgid


def gpu_worker(jobs, gpu_id, queue):
    device = f'cuda:{gpu_id}'
    for job in jobs:
        pgid = process_book(job, device)
        queue.put(pgid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--text-dir', default='data/text')
    parser.add_argument('--output', default='smaller_corpus')
    parser.add_argument('--gpus', type=str, default=None, help='e.g. 0,1,2,3')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.corpus)
    print(f'Loaded {len(df)} entries')

    # build jobs
    jobs = []
    seen = {}
    for _, row in df.iterrows():
        pgids = get_pgids(row['PG'])
        if not pgids:
            continue
        title = str(row.get('Title', row.get('title', '')))
        base = sanitize(title) if title else 'novel'

        for i, pgid in enumerate(pgids):
            in_path = Path(args.text_dir) / f'PG{pgid}_text.txt'
            if not in_path.exists():
                print(f'Missing: {in_path}')
                continue

            if len(pgids) > 1:
                folder = f'{base}{i+1}'
            else:
                key = base.lower()
                if key in seen:
                    seen[key] += 1
                    folder = f'{base}_{seen[key]}'
                else:
                    seen[key] = 1
                    folder = base

            out_dir = Path(args.output) / folder
            jobs.append((pgid, title, str(in_path), str(out_dir)))

    # resume
    if args.resume:
        before = len(jobs)
        jobs = [j for j in jobs if not list(Path(j[3]).glob('*.tokens'))]
        print(f'Resume: {before - len(jobs)} done, {len(jobs)} left')

    if not jobs:
        print('Nothing to do')
        return

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f'Processing {len(jobs)} books')

    # multi-gpu
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        chunks = [[] for _ in gpu_ids]
        for i, job in enumerate(jobs):
            chunks[i % len(gpu_ids)].append(job)

        queue = mp.Queue()
        procs = []
        for gid, chunk in zip(gpu_ids, chunks):
            if chunk:
                p = mp.Process(target=gpu_worker, args=(chunk, gid, queue))
                p.start()
                procs.append(p)

        done = 0
        total = len(jobs)
        while done < total:
            pgid = queue.get()
            done += 1
            print(f'[{done}/{total}] PG{pgid} done')

        for p in procs:
            p.join()

    # single device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i, job in enumerate(jobs):
            pgid, title, _, _ = job
            print(f'[{i+1}/{len(jobs)}] PG{pgid}: {title[:40]}')
            process_book(job, device)

    print('Done')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()