# How to Run



## MLFlow

I have created MLFlow to experiment changing the model to track how it's changed

This will allow me to promote the best model to production

## CI/CD pipeline

I've used github actions to implement pylint scoring and running of the tests - this doesn't stop the PR happening but it sends a warning to my email

### Original Trial

The accuracy on train dataset is:  0.9815
The accuracy on test dataset is:  0.959

F1 score is:  0.8624
Precision is:  0.9698
Recall is:  0.7764

This is overfitting

## With Stratify 

The accuracy on train dataset is:  1.0
The accuracy on test dataset is:  0.9635

F1 score is:  0.8777
Precision is:  0.9493
Recall is:  0.8162


This is overfitting - still slightly better though 

## Reducing Max Depth

The accuracy on train dataset is:  0.9606
The accuracy on test dataset is:  0.957

F1 score is:  0.8576
Precision is:  0.9152
Recall is:  0.8069


**I then added alpha and lambda dampening** 


# How I would deploy this in Azure / Databricks 

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

    * Then deploy it - using predefined things

6. We can then view the endpoint - where it would be a rest endpoint which is a URL we can send data to
7. Test the model by sending data to it  using ` service.scoring_uri ` - you then get your output back


# Question Answering

### What are the assumptions you have made for this service and why?

* The training data is representative of the overall customer base - if it's not representative then our predictions may over / under predict the percentage of claiming %

* The database table we're pulling from is clean / correct, will remain updated and the data types won't change.
This is because if the data is not accurate then the model results can't be. If the data types change then this will mean the model may break unless there's automated downstream updating of them


###  What considerations are there to ensure the business can leverage this service?

* Downtime - we need to ensure that it's set up in a way where new outputs can be produced 24/7
This means we need error handling and that the testing in CI/CD process needs to be comprehensive

* The business understands the model - if the business don't understand the model then they can't trust it and won't use it.
We therefore need to involve the stakeholders in feature selection and setup time to explain the model outputs, where we'd answer all their questions.

* Intergration with existing situations - we need to automate the downstream and perhaps the upstream.
We also need to read the unseen data and output the results  to the correct location reliably and automatically.
Also, if we're doing automatic retraining of the model then we'd need the model integrated into the same location as where the training data would come from.

* Scalability - we need to ensure that if there's busy periods when lots of applications come in then the solution can handle it




### Which traditional teams within the business would you need to talk to and why?

* Data Engineering / Data Science / MLOPS / Devops

Our team would manage the Data Science and MLOPS and as much data engineering as required

We would need to work with the central data engineering team to ensure the tables they maintain are kept up to date 

* Cloud Infrastucture

They would ensure that the Azure environment is kept running reliably in an efficient way.

They would need to inform us of any outages.

They may need to build new infrastructure for us.

* Data Governance / Compliance

They'll ensure that we're using the data responsibly and aren't using features that we shouldn't be 

* Trading team 

We need to gain their trust by explaining the model to them.

Otherwise they won't risk using it

We can also use their business expertise to consider parameters to the model for example

* Cyber Security

They'll ensure that the endpoint and the data is secure


### What is in and out of scope for your responsibility?

**In**



* Model developlement - building, training and evaluating

* Tuning the model

* Deploying the model

* Model tracking with MLflow 

* Tracking Drift - both in terms of concept drift and data drift

* Deciding where to retrain - where if the updated data changes the results we'll have to explain that to the business

**Out**

* Integration wiith current systems

* Building a sophisticated front end - perhaps we could make a streamlit page though 

* Maintance of the pipeline to keep the table we read from up to date