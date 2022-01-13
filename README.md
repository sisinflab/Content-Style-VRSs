# Leveraging Content-Style Item Representation for Visual Recommendation

<figure>
    <img src="https://github.com/sisinflab/Content-Style-VRSs/blob/master/FinalModel.png"/>
</figure>

This is the official GitHub repository for the paper [**Leveraging Content-Style Item Representation for Visual Recommendation**](https://www.researchgate.net/publication/356541933_Leveraging_Content-Style_Item_Representation_for_Visual_Recommendation), accepted as short paper at the 44th European Conference on Information Retrieval (ECIR 2022).

**Authors:** Yashar Deldjoo, Tommaso Di Noia, Daniele Malitesta*, Felice Antonio Merra.
<br>\**corresponding author*

## Disclaimer \#1
The code for the proposed model and the baselines was implemented in of **Elliot**. If you want to reproduce the exact same results described in the paper, please consider this essential version of Elliot. However, please, refer to the most updated version of the official [GitHub page](https://github.com/sisinflab/elliot) and the [documentation](https://elliot.readthedocs.io/en/latest/) for detailed information on how to run the framework.

## Disclaimer \#2
We are working to implement the proposed model within the most recent version of Elliot so that we can take advantage of all the new introduced features. When this happens, we will remove every script from here, and directly link to Elliot.

### Dataset
At this anonymized [link](https://drive.google.com/file/d/1v1XeDlpYAwod3jfIutD9zS_ct9Q3aTgB/view?usp=sharing) you may find the datasets adopted in the paper. For each item image, we have already provided the extracted visual features required for all visual-based baselines and our proposed model. Please, just put the downloaded datasets into the ```./data/``` folder.

### Training and evaluating the models
- For each experiment, please run the following script:
```
python -u sample_main.py --type_experiment [NAME OF EXPERIMENT] --dataset [DATASET NAME]
```
where the parameters ```type_experiment``` and ```dataset``` may be set using these values:

- ```type_experiment: SEE EXAMPLE BELOW```
- ```dataset: {amazon_men, amazon_boys_girls}```

This will train or test all the models on the considered datasets, following the configuration files in ```./config_files/``` (where you can find a detailed overview on all explored values for all the hyperparameters). Note that, the ```type_experiment``` field has to be completed with the name of the configuration file without the dataset name, e.g., if the file is named ```evaluate_amazon_men.yml```, then ```type_experiment: evaluate```. The configuration files ```evaluate_amazon_men.yml``` and ```evaluate_amazon_boys_girls.yml``` allow to reproduce all results.

### Our proposed method
You may find the scripts for our proposed method at the path ```./elliot/recommender/custom/```, while the corresponding data samplers can be found at the path ```./elliot/dataset/dataloaders/```.

### Reproducibility Details

We randomly initialize the model parameters of all tested methods with a Gaussian distribution with a mean of 0 and standard deviation of 0.01 and set the latent factor dimension to 128 following the experimental settings used in [Chen et al.](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Attentive%20Collaborative%20Filtering%20Multimedia%20Recommendation%20with%20Item-%20and%20Component-Level%20Attention.pdf). We explore the following hyperparameters via grid-search: the learning rate in {0.0001, 0.001, 0.01} and the regularization coefficients in {0.00001, 0.001}, whereas we fix the batch size to 256, and the temperature to 10.0. We adopt early-stopping to avoid overfitting and choose the best model configuration for each algorithm according to the hit ratio (HR), i.e., the validation metric, as in [Chen et al.](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Attentive%20Collaborative%20Filtering%20Multimedia%20Recommendation%20with%20Item-%20and%20Component-Level%20Attention.pdf).
