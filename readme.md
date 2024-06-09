# Overall  notes

## Architecture

* end_to_end.py - where the model is ran
* end_to_end_utils.py - where callable functions are stored
* requirements.txt - where packages that need to be installed are defined
* test_my_module.py - where the tests are stored

## MLFlow

I have created MLFlow to try out changes in the model to track how it's changed

This will allow me to promote the best model to production, based on the tracked features.

Notes on the models I've ran are in this document.

## CI/CD pipeline

I've created two github actions workflows to run each time I push code to main

* Pylint - This gives the code a score out of 10 based on things such as not unnecessarily importing things and functions being properly defined.

* Run Test - this ensures that the tests written in test_my_module.py are checked using pytest

These send an alert if they fail - where I would change this to not allowing it to merge in future.

## Runs

### Original Trial

* The accuracy on train dataset is:  0.9815
* The accuracy on test dataset is:  0.959
* F1 score is:  0.8624
* Precision is:  0.9698
* Recall is:  0.7764

This is overfitting as the accuracy if the train > accuracy of the test

### With Stratify = y

* The accuracy on train dataset is:  1.0
* The accuracy on test dataset is:  0.9635

* F1 score is:  0.8777
* Precision is:  0.9493
* Recall is:  0.8162


This is still overfitting.

I next tried reducing the maximum defined max depth from 8 to 5 to combat this

### Reducing Max Depth

* The accuracy on train dataset is:  0.9606
* The accuracy on test dataset is:  0.957
* F1 score is:  0.8576
* Precision is:  0.9152
* Recall is:  0.8069

This has made both training and test accuracies worse so this is removed.

### Adding alpha and lambda regularization

* The accuracy on train dataset is: 0.9744
* The accuracy on the test dataset is: 0.96
* F1 score is 0.8671
* Precision score is 0.9631
* Recall score is 0.7885

### Alpha and lambda regularization with stratify=y

* The accuracy on train dataset is: 0.9978
* The accuracy on the test dataset is: 0.963
* F1 score is 0.8763
* Precision score is 0.9458
* Recall score is 0.8162

## Determining which model to use

When promoting to production I would use the 2nd model. This is because the f1 score and the test accuracy are the highest.

The test accuracy is lower than the training accuracy so there is still overfitting.

This has got stratify=y, meaning there is a representative sample of people who claim and don't claim in the test data.

I wouldn't include alpha or lambda regularization and not change the tree depth.

## How I would deploy this in Azure / Databricks 

1. As done I would create, train and evaluate the model
2. I'd then run multiple experiments in MLFlow until I found a suitable model
3. I'd then register the model in ML Flow within Azure so it can be reused

```
model_uri = f"runs:/{run.info.run_id}/model"
    model_name = "credit_model"
    mlflow.register_model(model_uri=model_uri, name=model_name) 

```

4. This then has version control and you can then see how the different versions performs within jobs

5. Set up for deployment

    * We set up a curated environment - this can be ubuntu - this can track your packages for example
    * Defining the deployment config - this defines whether it's an Azure Web service for example, the cpu_cores, memory, tags and descriptions


6. Deployment

    * Define the registered model

    * Create an inference_config - using the predifined environment

    * Then deploy it - using predefined parameters

6. We can then view the endpoint - where it would be a rest endpoint which is a URL we can send data to
7. Test the model by sending data to it  using ` service.scoring_uri ` - you then get your output back


## Question Answering

### What are the assumptions you have made for this service and why?

* The training data is representative of the overall customer base - if it's not representative then our predictions may over / under predict the percentage of claiming %

* The database table we're pulling from is clean / correct, will remain updated and the data types won't change.
This is because if the data is not accurate then the model results can't be. If the data types change then this will mean the model may break unless there's automated downstream updating of them.


###  What considerations are there to ensure the business can leverage this service?

* Downtime - we need to ensure that it's set up in a way where new outputs can be produced 24/7.
This means we need error handling and that the testing in CI/CD process needs to be comprehensive.

* The business understands the model - if the business don't understand the model then they can't trust it and won't use it.
We therefore need to involve the stakeholders in feature selection and setup time to explain the model outputs, where we'd answer all their questions.

* Intergration with existing situations - we need to read the unseen data and output the results  to the correct location reliably and automatically.
Also, if we're doing automatic retraining of the model then we'd need the model integrated into the same location as where the training data would come from.

* Scalability - we need to ensure that if there's busy periods when lots of applications come in then the solution can handle it




### Which traditional teams within the business would you need to talk to and why?

* Data Engineering / Data Science / MLOPS - Our team would manage the Data Science and MLOPS and as much data engineering as required. We would need to work with the central data engineering team to ensure the tables they maintain are kept up to date 

* Cloud Infrastucture - They would ensure that the Azure environment is kept running reliably in an efficient way. They would also need to inform us of any outages and may need to build new infrastructure for us.

* Data Governance / Compliance - They'll ensure that we're using the data responsibly and aren't using features that we shouldn't be.

* Trading team  - We need to gain their trust by explaining the model to them, otherwise they won't risk using it. We can also use their business expertise to consider parameters to the model for example.

* Cyber Security - They'll ensure that the endpoint and the data is secure.


### What is in and out of scope for your responsibility?

**In**



* Model developlement - building, training and evaluating

* Tuning the model

* Deploying the model

* Model tracking with MLflow 

* Tracking Drift - both in terms of concept drift and data drift.

* Deciding where to retrain - where if the updated data changes the results we'll have to explain that to the business.

**Out**

* Integration with current systems

* Building a sophisticated front end - perhaps we could make a streamlit page though 

* Maintance of the pipeline to keep the table we read from up to date


## Next steps

* Changing from a classifier to a regressor - this is high priority as it's needed for Trevor's requirement.
This would mean we'd need to change how we evaluate the model by using MAPE for example.

* Defining package versions in requirements.txt - this means that if we don't risk the package updating changing the results or not being compatible with another package

* Adding artificial data into the pipeline to balance the people who are going to claim and aren't going to - this could be done using smote once dummies are made for all the categorical variables.

* Further data checks / tests - i've checked that for the claim_status column the max is 1 and the min is 0. I would add checks to other columns such as ensuring age is sensible

* Improve CI/CD pipeline - I would add further tests and run black on this
* Migrate to Databricks / Azure - following the steps outlined earlier is this document
* Ensemble method - I would run multiple machine learning models at the same time, then use ensemble to choose the best one
