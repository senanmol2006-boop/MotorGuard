# Motor Guard

### Dataset Acknowledgements
This dataset used for this model is self sampled.

### Description
The project uses simple and structured binary approach in predicting income range (<=50K or >50K) of an individual using Machine Learning and FastAPI.

The prediction is carried out based on the following input data from the user:

- current
- vibration (0 --> baseline vibration & 1 --> irregular vibration)

The given model uses RandomForest Classification as well as Artificial Neural Network to train the model.

- Artificial Neural Network : predicts the overall health score of the given motor based on the live feed of the data.
- RandomForest : performs multi variable classification to predict the type of fault in the motor

Thank you!
