import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import numpy as np  
import warnings
from fastapi.responses import HTMLResponse

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create a FastAPI app instance
app = FastAPI()

# Load the pickled model using a relative file path
with open('decision_tree_model.pkl', "rb") as model_file:
    model = pickle.load(model_file)

html_content = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson's Disease Detection</title>
    <style>
        /* Add your CSS styles here */
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }
        
        .form-container {
            text-align: center;
        }
        
        h1 {
            color: #333;
        }
        
        form {
            margin-top: 20px;
        }
        
        input[type='number'],
        input[type='text'] {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }
        
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0, 0, 0, 0.4);
            padding-top: 100px;
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            border-radius: 10px;
            position: relative;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .close {
            color: #aaa;
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        /* Animation for disease */
        @keyframes shake {
            10%, 90% {
                transform: translate3d(-1px, 0, 0);
            }
            
            20%, 80% {
                transform: translate3d(2px, 0, 0);
            }

            30%, 50%, 70% {
                transform: translate3d(-4px, 0, 0);
            }

            40%, 60% {
                transform: translate3d(4px, 0, 0);
            }
        }

        .disease-animation {
            animation: shake 0.5s;
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="form-container">
            <img src="https://clubrunner.blob.core.windows.net/00000050114/Images/TRF%20Logos/AOF_disease_color_side_title_RGB_EN.png" alt="Logo" width="300">
            <h1>Parkinson's Disease Detection</h1>
            <form id="predictionForm">
                <label for="name">Name and Surname:</label>
                <input type="text" id="name" name="name" required><br><br>
                <label for="PPE">PPE:</label>
                <input type="number" id="PPE" name="PPE" step="any" required><br><br>
                <label for="MDVPFo">MDVP:Fo(Hz):</label>
                <input type="number" id="MDVPFo" name="MDVPFo" step="any" required><br><br>
                <label for="MDVPJitter">MDVP:Jitter(%):</label>
                <input type="number" id="MDVPJitter" name="MDVPJitter" step="any" required><br><br>
                <label for="MDVPRAP">MDVP:RAP:</label>
                <input type="number" id="MDVPRAP" name="MDVPRAP" step="any" required><br><br>
                <label for="D2">D2:</label>
                <input type="number" id="D2" name="D2" step="any" required><br><br>
                <button type="submit" style="font-size: 18px;">Predict</button>
            </form>
        </div>
        <!-- Prediction Result Modal -->
        <div id="predictionModal" class="modal">
            <div class="modal-content">
                <span class="close" id="closeModal">&times;</span>
                <img src="https://clubrunner.blob.core.windows.net/00000050114/Images/TRF%20Logos/AOF_disease_color_side_title_RGB_EN.png" alt="Logo" width="190" style="margin: 0 auto;">
                <h2 style="font-size: 30px;">Prediction Result</h2>
                <p id="predictionResult" style="font-size: 20px;"></p>
                <!-- Buttons inside the modal -->
                <div style="text-align: center;">
                    <button id="refreshButton" style="font-size: 18px;">Refresh</button>
                    <button id="treatmentButton" style="font-size: 18px;">View Treatment</button>
                </div>
            </div>
        </div>
        <!-- Treatment Modal -->
        <div id="treatmentModal" class="modal">
            <div class="modal-content">
                <span class="close" id="closeTreatmentModal">&times;</span>
                <h2>Current Medical Treatments for Parkinson's Disease</h2>
                <p>Introduction:</p>
                <p>Parkinson's disease (PD) is a progressive neurological disorder characterized by motor and non-motor symptoms resulting from the degeneration of dopamine-producing neurons in the brain. While there is currently no cure for Parkinson's disease, several medications are available to help manage symptoms and improve quality of life for affected individuals. This treatise provides an overview of the primary medications used in the treatment of Parkinson's disease.</p>
                <h3>1. Levodopa:</h3>
                <p>- Levodopa is the most effective medication for managing the motor symptoms of Parkinson's disease.
Levodopa is converted into dopamine in the brain, replenishing depleted dopamine levels and improving motor function.
Common formulations of levodopa include levodopa/carbidopa (Sinemet) and levodopa/benserazide (Madopar).
Side effects may include nausea, vomiting, dyskinesias (involuntary movements), and motor fluctuations (wearing off or on-off phenomena) with long-term use.</p>
                <h3>2. Dopamine Agonists:</h3>
                <p>- Dopamine agonists mimic the action of dopamine in the brain, stimulating dopamine receptors and alleviating motor symptoms.
Examples of dopamine agonists include pramipexole (Mirapex), ropinirole (Requip), and rotigotine (Neupro).
Dopamine agonists can be used as monotherapy or in combination with levodopa.
Side effects may include nausea, dizziness, hallucinations, compulsive behaviors (such as gambling or shopping), and orthostatic hypotension.</p>
                <h3>3. Monoamine Oxidase Type B (MAO-B) Inhibitors:</h3>
                <p>- MAO-B inhibitors work by inhibiting the enzyme monoamine oxidase type B, which breaks down dopamine in the brain, thereby increasing dopamine levels.
Rasagiline (Azilect) and selegiline (Eldepryl, Zelapar) are examples of MAO-B inhibitors used in the treatment of Parkinson's disease.
MAO-B inhibitors can be used alone or in combination with levodopa.
Side effects may include insomnia, headache, and gastrointestinal disturbances.</p>
                <h3>4. Catechol-O-Methyltransferase (COMT) Inhibitors:</h3>
                <p>- COMT inhibitors prolong the duration of levodopa's effect by inhibiting the enzyme catechol-O-methyltransferase, which breaks down levodopa in the bloodstream.
Entacapone (Comtan) and tolcapone (Tasmar) are COMT inhibitors used in conjunction with levodopa/carbidopa.
Side effects may include diarrhea, dyskinesias, and orange discoloration of urine.</p>
                <h3>5. Anticholinergics:</h3>
                <p>- Anticholinergic medications help alleviate tremors and rigidity by blocking the action of acetylcholine, a neurotransmitter involved in muscle control.
Trihexyphenidyl (Artane) and benztropine (Cogentin) are commonly used anticholinergic drugs in the treatment of Parkinson's disease.
Anticholinergics are typically reserved for younger patients with prominent tremors and minimal cognitive impairment due to their potential for cognitive side effects.</p>
                <br></br>
                <br></br>
                <br></br>
                <!-- Add more treatments here-->       
                </div>
        </div>
    </div>

    <script>
        // Get the modal
        const modal = document.getElementById('predictionModal');
        const treatmentModal = document.getElementById('treatmentModal');

        // Get the <span> element that closes the modal
        const closeModal = document.getElementById('closeModal');
        const closeTreatmentModal = document.getElementById('closeTreatmentModal');

        // When the user clicks on <span> (x) or submits the name form, close the modal
        closeModal.onclick = function() {
            modal.classList.add('fadeOut');
            setTimeout(() => {
                modal.style.display = "none";
                modal.classList.remove('fadeOut');
            }, 500);
        };

        closeTreatmentModal.onclick = function() {
            treatmentModal.classList.add('fadeOut');
            setTimeout(() => {
                treatmentModal.style.display = "none";
                treatmentModal.classList.remove('fadeOut');
            }, 500);
        };

        // When the user clicks anywhere outside of the modal, close it
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.classList.add('fadeOut');
                setTimeout(() => {
                    modal.style.display = "none";
                    modal.classList.remove('fadeOut');
                }, 500);
            } else if (event.target == treatmentModal) {
                treatmentModal.classList.add('fadeOut');
                setTimeout(() => {
                    treatmentModal.style.display = "none";
                    treatmentModal.classList.remove('fadeOut');
                }, 500);
            }
        };

        const form = document.getElementById('predictionForm');
        const refreshButton = document.getElementById('refreshButton');
        const treatmentButton = document.getElementById('treatmentButton');

        refreshButton.onclick = function() {
            // Reload the page to refresh
            location.reload();
        };

        treatmentButton.onclick = function() {
            // Show the treatment modal
            treatmentModal.style.display = "block";
        };

        form.addEventListener('submit', async(event) => {
            event.preventDefault();

            const formData = new FormData(form);
            const formValues = Object.fromEntries(formData.entries());

            const response = await fetch('/predict/?' + new URLSearchParams(formValues));
            const data = await response.json();
            const predictionResult = document.getElementById('predictionResult');
            const name = formValues["name"];
            if (data.prediction === 1) {
                predictionResult.textContent = `${name}, you are diagnosed with Parkinson's disease.`;
                modal.classList.add('disease-animation');
            } else {
                predictionResult.textContent = `${name}, you are not diagnosed with Parkinson's disease.`;
            }

            // Show the modal with animation
            modal.style.display = "block";

            // Remove animation class after animation completes
            setTimeout(() => {
                modal.classList.remove('disease-animation');
            }, 500);
        });
    </script>
</body>

</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)


@app.get("/predict/")
async def predict(PPE: float = Query(..., title="PPE"),
                  MDVPFo: float = Query(..., title="MDVPFo"),
                  MDVPJitter: float = Query(..., title="MDVPJitter"),
                  MDVPRAP: float = Query(..., title="MDVPRAP"),
                  D2: float = Query(..., title="D2")):

    # Create a DataFrame from user input
    user_input = pd.DataFrame([[PPE, MDVPFo, MDVPJitter, MDVPRAP, D2]],
                               columns=['PPE', 'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:RAP', 'D2'])

    # Make predictions using the model
    prediction = model.predict(user_input)

    # Convert prediction result to native Python type
    prediction = int(prediction)  # or float(prediction)

    # Return the prediction result
    return {"prediction": prediction}


# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="192.168.1.13",
        port=5000,
        log_level="debug",
    )
