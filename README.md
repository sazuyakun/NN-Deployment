## Simple implementation

1. Create an environment
```bash
conda create --prefix venv
```

2. Activate the environment
```bash
conda activate ./venv
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. Train the model
```bash
python app/simple_neural_network.py
```

5. Run the API server
```bash
python app/api_server.py
```

## Docker Implementation


1. Build the Docker image
```bash
docker build -t simple_nn .
```

2. Run the container
```bash
docker run --name simple_nn_container -p 8000:8000 simple_nn
```

---

## Check API endpoints

### Using curl

1. Test basic endpoint
```bash
curl http://localhost:8000/
```

2. Test prediction endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.5, 1.3]}'
```

### Using Postman

1. Set a POST request
2. Enter this URL:
```
http://localhost:8000/predict
```
3. Body → raw → JSON → Enter features:
```json
{
  "features": [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.5, 1.3]
}
```
