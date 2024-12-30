# House_prediction_model
House Price Prediction Model
Welcome to the House Price Prediction Model repository! This project uses the California Housing Dataset to predict median house values based on various features. The model is built using Python, Scikit-learn, Pandas, and Streamlit, and serves as a demonstration of machine learning and web application development skills.

🚀 Project Overview
The House Price Prediction Model project includes:

Exploratory Data Analysis (EDA): Understand the dataset and visualize relationships between features and target values.
Model Training: Train a regression model to predict house prices.
Model Evaluation: Evaluate the performance of the model using metrics like Mean Squared Error (MSE) and R² score.
Deployment: Deploy the model using Streamlit for a simple and interactive user interface.
This project is part of my portfolio to demonstrate proficiency in machine learning, data visualization, and deployment.

🛠️ Tools and Technologies
Python: Programming language used for model implementation.
Pandas & NumPy: For data manipulation and analysis.
Scikit-learn: For building and evaluating the regression model.
Matplotlib & Seaborn: For creating insightful data visualizations.
Streamlit: For deploying the machine learning model with a web-based interface.
📁 Dataset
The project uses the California Housing Dataset provided by Scikit-learn.

Dataset Description: Predicts median house value (MedHouseValue) based on features like median income, average number of rooms, and proximity to employment hubs.
Source: California Housing Dataset
📊 Key Features
Data Preprocessing:

Loaded dataset into a Pandas DataFrame.
Explored dataset with california.DESCR and visualized relationships using scatterplots.
Exploratory Data Analysis:

Visualized trends in housing prices based on features like MedInc, AveRooms, and HouseAge.
Cleaned and split data for training and testing.
Model Training:

Built a Linear Regression model using Scikit-learn.
Extracted feature weights to interpret model predictions.
Model Evaluation:

Evaluated model accuracy using Mean Squared Error (MSE) and R² score.
Achieved interpretable results highlighting feature impacts.
Interactive Deployment:

Deployed the model as a web app using Streamlit.
Users can input custom feature values to predict house prices.
💻 Installation
To run the project locally:

Clone the repository:

bash
Copy code
git clone https://github.com/qaysSE/house_prediction_model.git
cd house_prediction_model
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run Hajibrahim_House_price_pred_model.py
Open your browser at the URL provided (e.g., http://localhost:8501/).

📈 Results
Model Performance:

Mean Squared Error (MSE): [0.40]
R² Score: [80%]
Feature Importance:

MedInc (Median Income): Most impactful feature.
Other features contribute less significantly but provide meaningful context.
🎯 Future Enhancements
Add feature engineering for improved model accuracy.
Experiment with other regression algorithms like Random Forest or XGBoost.
Improve the Streamlit interface for better user experience.
📂 Repository Structure
plaintext
Copy code
house-price-prediction/
├── data/                     # Dataset or data preprocessing files
├── Hajibrahim_House_price_pred_model.py  # Main script for training and deployment
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── visuals/                  # Data visualizations and screenshots
📸 Screenshots
EDA Scatterplot	Streamlit App
🤝 Contribution
Feel free to fork this repository, raise issues, or submit pull requests to improve the project.

📧 Contact
For questions or collaborations, feel free to reach out:
Name: Qays Hajibrahim
Email: qayshajibrahim@gmail.com
GitHub: qaysSE

