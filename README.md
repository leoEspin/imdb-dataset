# LSTM model for the IMDB dataset

Training, deployment and testing code for a sentiment analysis model trained on the IMDB dataset.

**Author**: Leonardo Espín

**Date**: 4/25/2021

This repo uses a containerized workflow in Google Cloud to train the model. In a VM or cloud shell:

* Build the container by running

    `docker build  -t lstm .`
* Test the container locally by running

    ```
    docker run lstm --job-dir=gs://$BUCKET/lstm_model/ \
        --epochs=1 \
        --max-seq-len=25
    ```

    this will limit the text-sequence size to 25 words, to speed up the test training
* Push the container to your container registry

    ```bash
    docker image tag lstm gcr.io/$MY_PROJECT/lstm
    docker push gcr.io/$MY_PROJECT/lstm
    ```
* Submit the training job to the AI Platform using the default model arguments. The configuration file requests a Tesla T4 GPU

	```bash
	gcloud ai-platform jobs submit training lstm_$(date +"%Y%m%d_%H%M%S") \
      --job-dir gs://$BUCKET/lstm_model/ \
      --region $REGION \
      --master-image-uri gcr.io/$MY_PROJECT/lstm \
      --config config/config.yaml
	```
	
## Testing

The trained model can be tested through the `test.ipynb` notebook in the test folder. Tests include making predictions through a rest api using the `tensorflow/serving` container.