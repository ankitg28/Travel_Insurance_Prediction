# Travel_Insurance_Prediction
Please refer to my medium article explaining this project in detail -> <a href="https://medium.com/@goyalankit28/travel-insurance-prediction-journey-from-dataset-selection-to-ui-based-prediction-44eeb996f778">Travel Insurance Prediction: Journey from dataset selection to UI Based Prediction</a><br><br>

Through this project I want to explain the whole process of creating a simple data science project right from selecting a dataset for prediction to creating a model to finally deploying on a server to have our model predict based on User-based feature inputs. To keep this content short, I have covered all the steps in brief but feel free to refer to my Jupyter Notebook to follow along and dive deeper into every step of the process. Hope you enjoy reading my project!

<img src="https://github.com/ankitg28/Travel_Insurance_Prediction/blob/main/Flow.jpg" alt="Travel_Insurance_Prediction">

About the Dataset<br>
Atour & travels company is offering travel insurance package to their customers. The new insurance package also includes covid cover. The company wants to know which customers would be interested to buy it based on their database history. The insurance was offered to some of the customers in 2019 and the given data has been extracted from the performance/sales of the package during that period. The data is provided for almost 2000 of its previous customers and we are required to build an intelligent model that can predict if the customer will be interested to buy the travel insurance package.

Kaggle Dataset Link: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data

Column Description for our Dataset
1. Target Variable/Dependent Variable

1.1 TravelInsurance — Did the customer buy travel insurance package during introductory offering held in the year 2019. This is the variable we have to predict

2. Predictor Variables/Independent Variables

2.1 Age — Age of the customer<br>
2.2 Employment Type — The sector in which customer is employed<br>
2.3 GraduateOrNot — Whether the customer is college graduate or not<br>
2.4 AnnualIncome — The yearly income of the customer in indian rupees[rounded to nearest 50 thousand rupees]<br>
2.5 FamilyMembers — Number of members in customer’s family<br>
2.6 ChronicDiseases — Whether the customer suffers from any major disease or conditions like Diabetes/high BP or Asthma, etc<br>
2.7 FrequentFlyer — Derived data based on customer’s history of booking air tickets on at-least 4 different instances in the last 2 years[2017–2019]<br>
2.8 EverTravelledAbroad — Has the customer ever travelled to a foreign country<br>

<img src="https://miro.medium.com/max/4800/1*rC9ytBFXh6ncHcdX99qZNw.gif" alt="Travel_Insurance_Prediction">
