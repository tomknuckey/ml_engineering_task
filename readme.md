# Overall Notes

## Architecture

* `end_to_end.py` - where the model is run
* `end_to_end_utils.py` - where callable functions are stored
* `requirements.txt` - where packages that need to be installed are defined
* `test_my_module.py` - where the tests are stored

## MLFlow

I have created MLFlow to try out changes in the model and track how it changes.

This will allow me to promote the best model to production based on the tracked features.

Notes on the models I've run are in this document.

## CI/CD Pipeline

I've created two GitHub Actions workflows to run each time I push code to the main branch:

* **Pylint** - This gives the code a score out of 10 based on factors such as avoiding unnecessary imports and ensuring functions are properly defined.

* **Run Test** - This ensures that the tests written in `test_my_module.py` are checked using `pytest`.

These send an alert if they fail; in the future, I would change this to not allowing merges if they fail.

## Runs

### Original Trial

* The accuracy on the train dataset is: 0.9815
* The accuracy on the test dataset is: 0.959
* F1 score is: 0.8624
* Precision is: 0.9698
* Recall is: 0.7764

This is overfitting as the accuracy of the train > accuracy of the test.

### With Stratify = y

* The accuracy on the train dataset is: 1.0
* The accuracy on the test dataset is: 0.9635
* F1 score is: 0.8777
* Precision is: 0.9493
* Recall is: 0.8162

This is still overfitting.

I next tried reducing the maximum defined max depth from 8 to 5 to combat this.

### Reducing Max Depth

* The accuracy on the train dataset is: 0.9606
* The accuracy on the test dataset is: 0.957
* F1 score is: 0.8576
* Precision is: 0.9152
* Recall is: 0.8069

This has made both training and test accuracies worse, so this change was removed.

### Adding Alpha and Lambda Regularization

* The accuracy on the train dataset is: 0.9744
* The accuracy on the test dataset is: 0.96
* F1 score is: 0.8671
* Precision score is: 0.9631
* Recall score is: 0.7885

### Alpha and Lambda Regularization with Stratify=y

* The accuracy on the train dataset is: 0.9978
* The accuracy on the test dataset is: 0.963
* F1 score is: 0.8763
* Precision score is: 0.9458
* Recall score is: 0.8162

## Determining Which Model to Use

When promoting to production, I would use the 2nd model. This is because the F1 score and the test accuracy are the highest.

The test accuracy is lower than the training accuracy, so there is still overfitting.

This model uses stratify=y, meaning there is a representative sample of people who claim and don't claim in the test data.

I wouldn't include alpha or lambda regularization or change the tree depth.

## How I Would Deploy This in Azure / Databricks

1. As done, I would create, train, and evaluate the model.
2. I'd then run multiple experiments in MLFlow until I found a suitable model.
3. I'd then register the model in MLFlow within Azure so it can be reused:

    ```python
    model_uri = f"runs:/{run.info.run_id}/model"
    model_name = "credit_model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    ```

4. This then has version control, and you can see how the different versions perform within jobs.
5. Set up for deployment:

    * We set up a curated environment - this can be Ubuntu - this can track your packages, for example.
    * Define the deployment config - this defines whether it's an Azure Web service, the CPU cores, memory, tags, and descriptions.

6. Deployment:

    * Define the registered model.
    * Create an `inference_config` - using the predefined environment.
    * Then deploy it - using predefined parameters.

7. We can then view the endpoint - where it would be a REST endpoint which is a URL we can send data to.
8. Test the model by sending data to it using `service.scoring_uri` - you then get your output back.

## Question Answering

### What Are the Assumptions You Have Made for This Service and Why?

* The training data is representative of the overall customer base - if it's not representative, then our predictions may over or under predict the percentage of claims.
* The database table we're pulling from is clean/correct, will remain updated, and the data types won't change. This is because if the data is not accurate, then the model results can't be. If the data types change, then this will mean the model may break unless there's automated downstream updating of them.

### What Considerations Are There to Ensure the Business Can Leverage This Service?

* **Downtime** - We need to ensure that it's set up in a way where new outputs can be produced 24/7. This means we need error handling and that the testing in the CI/CD process needs to be comprehensive.
* **Business Understanding** - If the business doesn't understand the model, then they can't trust it and won't use it. We, therefore, need to involve the stakeholders in feature selection and set up time to explain the model outputs, where we'd answer all their questions.
* **Integration with Existing Systems** - We need to read the unseen data and output the results to the correct location reliably and automatically. Also, if we're doing automatic retraining of the model, then we'd need the model integrated into the same location as where the training data would come from.
* **Scalability** - We need to ensure that if there are busy periods when lots of applications come in, then the solution can handle it.

### Which Traditional Teams Within the Business Would You Need to Talk to and Why?

* **Data Engineering/Data Science/MLOps** - Our team would manage the Data Science and MLOps and as much data engineering as required. We would need to work with the central data engineering team to ensure the tables they maintain are kept up to date.
* **Cloud Infrastructure** - They would ensure that the Azure environment is kept running reliably and efficiently. They would also need to inform us of any outages and may need to build new infrastructure for us.
* **Data Governance/Compliance** - They'll ensure that we're using the data responsibly and aren't using features that we shouldn't be.
* **Trading Team** - We need to gain their trust by explaining the model to them; otherwise, they won't risk using it. We can also use their business expertise to consider parameters for the model, for example.
* **Cyber Security** - They'll ensure that the endpoint and the data are secure.

### What Is In and Out of Scope for Your Responsibility?

**In Scope**

* Model development - building, training, and evaluating
* Tuning the model
* Deploying the model
* Model tracking with MLFlow
* Tracking drift - both in terms of concept drift and data drift.
* Deciding when to retrain - where if the updated data changes the results, we'll have to explain that to the business.

**Out of Scope**

* Integration with current systems
* Building a sophisticated front end - perhaps we could make a Streamlit page, though.
* Maintenance of the pipeline to keep the table we read from up to date.

## Next Steps

* Changing from a classifier to a regressor - this is high priority as it's needed for Trevor's requirement. This would mean we'd need to change how we evaluate the model by using MAPE, for example.
* Defining package versions in `requirements.txt` - this means that we don't risk the package updates changing the results or not being compatible with another package.
* Adding artificial data into the pipeline to balance the people who are going to claim and aren't going to - this could be done using SMOTE once dummies are made for all the categorical variables.
* Further data checks/tests - I've checked that for the `claim_status` column, the max is 1 and the min is 0. I would add checks to other columns such as ensuring age is sensible.
* Improve CI/CD pipeline - I would add further tests and run black on this.
* Migrate to Databricks/Azure - following the steps outlined earlier in this document.
* Ensemble method - I would run multiple machine learning models at the same time, and the best output would then be selected.
