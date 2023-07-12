import argparse
import glob
import logging
import os
import pandas as pd
import time
from app.scheduler import schedule
from definitions import OUT_DIR


def schedule_batch(files: list[str]):
    times = []
    scheduled = []
    for i, file in enumerate(files):
        start_time = time.time()
        scheduled.append(schedule(file))
        end_time = time.time()
        times.append(end_time - start_time)

        if i % 10 == 0:
            # output csv every 10 files, to avoid data missing.
            df = pd.DataFrame({"file": files[:i], "time": times, "is_scheduled": scheduled})
            df.to_csv(os.path.join(OUT_DIR, 'schedule_batch.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, type=str)
    args = parser.parse_args()

    assert os.path.isdir(args.dir), f"No such directory {args.dir}"

    files = glob.glob(os.path.join(args.dir, '*.json'))

    logging.basicConfig(level=logging.INFO)
    schedule_batch(files)
