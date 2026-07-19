# E-Commerce MLOps Pipeline: Session-Based Purchase Prediction

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict customer purchase behavior based on session interaction data. The project covers feature engineering, model training, evaluation, and deployment via a high-performance REST API.

##  Pipeline Architecture
Raw Session Data -> Feature Engineering -> Model Training & Selection (.pkl) -> FastAPI Service -> Docker Container


##  Project Structure
ecommerce/
├── models/
│   ├── model_a.pkl          # Trained Logistic Regression Model A
│   └── model_b.pkl          # Trained Random Forest Model B
├── feature_engineering.py
├── api.py                   # FastAPI Application for Real-Time Inference
├── requirements.txt         # Python Dependencies
└── Dockerfile               # Containerization Configuration


##  Features & Implementation
- **Feature Engineering:** Extracted session-based features including view counts, cart additions, unique product interactions, average prices, and session durations.
- **Model Training:** Implemented and compared classification models using `scikit-learn`.
- **Production API:** Built with `FastAPI` and `Pydantic` for data validation, ensuring rigid schema enforcement for incoming payloads.
- **Containerization Ready:** Fully configured with a lightweight `Dockerfile` for seamless deployment across any infrastructure.

## Local Setup (Without Docker)

1. **Install Dependencies:**
   pip install -r requirements.txt

2. **Run the FastAPI Server:**

   uvicorn api:app --reload

3. **Access Interactive Documentation:**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to interactively test the API models.

## Container Deployment (With Docker)

To build and run the application inside an isolated Docker container:


# Build the Docker image
docker build -t ecommerce-ml-api .

# Run the containerized service
docker run -d -p 8000:8000 ecommerce-ml-api


## API Specification

### Post Prediction via Model A
- **Endpoint:** `/predict/model_a`
- **Method:** `POST`
- **Payload Example:**
  json
  {
    "view_count": 30,
    "cart_count": 2,
    "unique_products": 1,
    "avg_price": 60,
    "session_duration": 3000
  }
  
- **Response Example:**
  json
  {
    "model_used": "Logistic_Regression_A",
    "will_purchase": 1,
    "purchase_probability": 0.85
  }
  