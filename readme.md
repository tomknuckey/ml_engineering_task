# How to Run



## MLFlow

I have created MLFlow to experiment changing the model to track how it's changed

This will allow me to promote the best model to production

### Original Trial

The accuracy on train dataset is:  0.9815
The accuracy on test dataset is:  0.959

This is overfitting

## With Stratify 

The accuracy on train dataset is:  1.0
The accuracy on test dataset is:  0.9635

This is overfitting - still slightly better though 

## Reducing Max Depth

The accuracy on train dataset is:  0.960625
The accuracy on test dataset is:  0.957

F1 score is:  0.8576158940397351
Precision is:  0.9151943462897526
Recall is:  0.8068535825545171


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