# paper-reader-source

`paper-reader-source` is a small source-collection subproject for `paper-reader`.

Current support:

- Hugging Face Daily Papers snapshot parsing
- A test script that fetches the current Daily Papers page and prints papers with more than 5 upvotes

## Run the Hugging Face test script

From the repo root:

```bash
python3 paper-reader-source/tests/test_huggingface_today.py
```

Optional flags:

```bash
python3 paper-reader-source/tests/test_huggingface_today.py --date 2026-04-10
python3 paper-reader-source/tests/test_huggingface_today.py --min-upvotes 10
python3 paper-reader-source/tests/test_huggingface_today.py --json
```

Notes:

- When `--date` is omitted, the script fetches `https://huggingface.co/papers` and uses the Daily Papers date exposed by Hugging Face on that landing page.
- The filter is strictly `> 5` upvotes by default, matching the original request.


## Hugging Face daily service

The subproject also includes a long-running scheduler service that:

- wakes up every day at `18:30` Beijing time (`Asia/Shanghai`)
- fetches the current Hugging Face Daily Papers page
- saves papers whose `upvotes >= 5`
- downloads each selected paper PDF into a per-day archive
- writes one daily manifest plus a `papers/` directory under `paper-reader-source/data/huggingface_daily/YYYY/MM/DD/`

Run it directly:

```bash
PYTHONPATH=paper-reader-source python3 -m paper_reader_source.service \
  --data-dir paper-reader-source/data/huggingface_daily
```

Run one immediate collection for testing:

```bash
PYTHONPATH=paper-reader-source python3 -m paper_reader_source.service \
  --data-dir paper-reader-source/data/huggingface_daily \
  --run-on-start --once
```

Tmux launcher:

```bash
paper-reader-source/scripts/run_huggingface_daily_service.sh
```
