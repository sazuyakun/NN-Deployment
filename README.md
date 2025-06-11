## Simple implementation

---

1. Create an environment
'''bash
conda create --prefix venv
'''

2. Activate the environment
'''bash
conda activate ./venv
'''

3. Install the required dependencies
'''bash
pip install -r requirements.txt
'''

4. Train the model
'''bash
python app/simple_neural_network.py
'''

5. Uncomment the last lines in the app/api_server.py and run
'''bash
python app/api_server.py
'''

---

## Docker Implementation

---

1. Run the command to build the dockerfile
'''bash
docker build -t simple_nn .
'''

2. Run the command to run the application in the container: simple_nn_container
'''bash
docker run --name simple_nn_container -p 8000:8000 simple_nn
'''

---

## Check API endpoints

### Postman

1. Set a POST request
2. Enter this as the URL
'''bash
http://localhost:8000/predict
'''
3. Body -> raw -> Enter any features value (should have 10 features)
'''bash
// example
{
  "features": [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.5, 1.3]
}
'''
