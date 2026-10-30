# Genetic Mutation Prioritization: Web + Android starter

This package converts the uploaded Streamlit prototype into a Flask website and JSON API, and includes a minimal Flutter Android client.

## What it does
- Website at `/`
- `POST /api/train` to download ClinVar, filter a gene, and train a Random Forest classifier
- `GET /api/status` to check training status
- `POST /api/predict` to predict using the currently trained in-memory model
- `GET /health` health check
- Flutter starter in `flutter_app/lib/main.dart`

## Important limitations
- Educational/research prototype only; not a clinical diagnostic tool.
- The app downloads the large ClinVar `variant_summary.txt.gz` file at runtime. First load can be slow and may exceed memory/time limits on some free hosting services.
- Training state is kept in memory. Restarting/redeploying the server clears the model and it must be trained again. Multiple server workers do not share that memory.
- Features are basic variant descriptors; this is not a full protein-structure model.
- Prediction scores are classifier scores, not clinically validated probabilities.

## Run locally on Windows
Use Python 3.11 or 3.12 for broad package compatibility.
1. Extract this ZIP.
2. Open Command Prompt in the extracted `genetic_mutation_web_and_android` folder.
3. Create a virtual environment:
   `py -3.12 -m venv .venv`
4. Activate it:
   `.venv\Scripts\activate`
5. Install dependencies:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
6. Start the website:
   `python app.py`
7. Open `http://127.0.0.1:5000`.

Internet is required to download ClinVar. If `py -3.12` is not installed, install Python 3.12 from python.org first.

## Deploy the website
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `Procfile`, `runtime.txt`, and the `templates` folder.
2. On Render, create a new Web Service and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 300`
5. Deploy, then open the URL Render provides.
6. Test `/health`, then open the home page.

Note: a large ClinVar download plus model training may exceed free-tier resource limits. If that happens, the robust next step is to prepare a smaller curated dataset offline and deploy it with the app.

## Flutter Android starter
1. Install Flutter and Android Studio.
2. Create a project in `flutter_app` by running `flutter create .` (from inside that folder) if it isn't already a Flutter project.
3. Replace `lib/main.dart` with the supplied file.
4. Set `apiBaseUrl` in `main.dart` to your deployed Render base URL (without a trailing slash).
5. Android app and website must use the same reachable API. `localhost` on a phone means the phone itself, not your PC.
6. Run `flutter run` to test, or `flutter build apk --release` to build an APK.

The included Flutter client is a starter UI. It can trigger training and submit demo features to the Flask API. Keep the deployed API base URL in the source when building the APK.
