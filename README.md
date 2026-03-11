# FinalProjectCS496
## Environment & API key setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a .env file in the root directory and fill in API key
```bash
OPEN_API_KEY=[API_KEY]
```

## Running the Project
Setup the rag database:
```bash
python rag_setup.py
```

Run the evaluation pipeline:
```bash
python evaluate.py
```

