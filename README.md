# paper-reader

A lightweight paper library and reader for PDF / Word papers.

## Features

- Batch upload PDF, `.doc`, and `.docx` files into any subfolder under the library root
- Automatically scan files already placed in the library by hand
- Extract paper titles from PDF / Word metadata/content when possible
- Search by extracted title and original filename
- Extract arXiv-style dates where possible and sort/group by date
- Read PDFs inline in the browser; preview `.docx`; open original `.doc` files directly
- Run on `0.0.0.0:8022`

## Library root

By default, the app scans and stores papers in:

- `docs/papers/`

That means your existing files in `docs/papers/` are included automatically.

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Start the server:

   ```bash
   .venv/bin/python run.py
   ```

3. Open the app in your browser:

   - `http://127.0.0.1:8022`
   - or from another machine on the same network: `http://<your-host>:8022`

## Notes

- `.pdf`: inline browser preview supported
- `.docx`: text preview supported via XML extraction
- `.doc`: legacy Word files are accepted and indexed; browsers usually open/download the original file instead of rendering it inline
- arXiv dates are extracted from text when possible; otherwise the app falls back to deriving month-level dates from arXiv IDs like `2501.12948`

## Tests

Run the smoke tests with:

```bash
.venv/bin/python -m unittest tests.test_app
```
