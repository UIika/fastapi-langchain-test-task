# FastAPI Text Summarizer

## Setup

1. Create a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

4. Test the endpoint:
   - Send a POST request to `http://127.0.0.1:8000/summarize` with a JSON body containing the text to be summarized.