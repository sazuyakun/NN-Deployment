## Simple implementation

---

1. Create an environment
'''
conda create --prefix venv
'''

2. Activate the environment
'''
conda activate ./venv
'''

3. Install the required dependencies
'''
pip install -r requirements.txt
'''

4. Train the model
'''
python app/simple_neural_network.py
'''

5. Uncomment the last lines in the app/api_server.py and run
'''
python app/api_server.py
'''

---

## Docker Implementation

---

1. Run the command to build the dockerfile
'''
docker build -t simple_nn .
'''

2. Run the command to run the application in the container: simple_nn_container
'''
docker run --name simple_nn_container -p 8000:8000 simple_nn
'''

---

## Check API endpoints

### Postman

1. Set a POST request
2. Enter this as the URL
'''
http://localhost:8000/predict
'''
3. Body -> raw -> Enter any features value (should have 10 features)
'''
// example
{
  "features": [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.5, 1.3]
}
'''
