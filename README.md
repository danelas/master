# AI Response Aggregator

A web application that queries multiple AI models with the same question and presents the responses in a user-friendly interface.

## Features

- Query multiple AI models (OpenAI, Anthropic, Google Gemini) simultaneously
- Compare responses side by side
- Simple and intuitive user interface
- Mock responses for testing without API keys

## Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file

5. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

## API Endpoints

- `GET /` - Health check
- `POST /query` - Send a question to multiple AI models

## License

MIT
