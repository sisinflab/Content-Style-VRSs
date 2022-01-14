# Leveraging Content-Style Item Representation for Visual Recommendation

<figure>
    <img src="https://github.com/sisinflab/Content-Style-VRSs/blob/master/FinalModel.png"/>
</figure>

This is the official GitHub repository for the paper [**Leveraging Content-Style Item Representation for Visual Recommendation**](https://www.researchgate.net/publication/356541933_Leveraging_Content-Style_Item_Representation_for_Visual_Recommendation), accepted as short paper at the 44th European Conference on Information Retrieval (ECIR 2022).

**Authors:** Yashar Deldjoo, Tommaso Di Noia, Daniele Malitesta*, Felice Antonio Merra.
<br>\**corresponding author*

If you want to use our model as baseline in your paper, please remember to cite us:

```
@misc{DDMM22,
  author       = "Deldjoo, Yashar and Di Noia, Tommaso and Malitesta, Daniele and Merra, Felice Antonio",
  title        = "Leveraging Content-Style Item Representation for Visual Recommendation",
  booktitle    = "44th European Conference on Information Retrieval",
  month        = "apr",
  year         = "2022",
  publisher    = "Springer",
  url          = "http://sisinflab.poliba.it/Publications/2022/DDMM22"
}
```

## Disclaimer \#1
The codes for the proposed model and the baselines were implemented in **Elliot**. If you want to reproduce the exact same results described in the paper, please consider this essential version of the framework. However, please, refer to the most updated version of the official [GitHub page](https://github.com/sisinflab/elliot) and the [documentation](https://elliot.readthedocs.io/en/latest/) for detailed information on how to run Elliot.

## Disclaimer \#2
We are working to integrate the proposed model within the most recent version of Elliot, so that we can take advantage of all the new introduced features. When this step is complete, we will have this repository directly linking to Elliot.

## Table of Contents
- [Requirements](#requirements)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [Baselines and Our Method](#baselines-and-our-method)
- [Reproducibility Details](#reproducibility-details)
- [Contacts](#contacts)

## Requirements

To begin with, please make sure your system has these installed:

* Python 3.6.8
* CUDA 10.1
* cuDNN 7.6.4

Then, install all required Python dependencies with the command:
```
pip install -r requirements.txt
```

## Training and Evaluation
For each experiment, please run the following script:
```
python start_experiment.py --config <configuration_filename_without_extension>
```

This will train or test all the models on the considered datasets, following the configuration files in ```./config_files/``` (where you can find a detailed overview on all explored values for all the hyperparameters).

## Datasets

|       Dataset      |   # Users   | # Items   |  # Interactions   |
| ------------------ | ------------------ | ------------------ | ------------------ |
|     [Amazon Boys & Girls](https://politecnicobari-my.sharepoint.com/:u:/g/personal/daniele_malitesta_poliba_it/EY7WwcUQHapLileGGlvW3iYBSCPf-WduNTVS_zc9j_sjTg?e=jm2kR8)*     |  1,425 | 5,019 | 9,213  |
|    [Amazon Men](https://politecnicobari-my.sharepoint.com/:u:/g/personal/daniele_malitesta_poliba_it/EdeOV-VDKeROtH4EJblo2XMBZm7HB4v4RkIAxxoor9PdQQ?e=PvcbPx)*    | 16,278 | 31,750 |  113,106  |

\* https://jmcauley.ucsd.edu/data/amazon/

The two adopted datasets are sub-categories of Amazon's product category *Clothing, Shoes and Jewelry*. We used the 2014 version (refer to the official link above). As for the filtering phase, we considered only the interactions recorded after 2010, and filtered out items and users with less than 5 interactions (applying the 5-core techniques on items and users). Finally, we applied the temporal leave-one-out to split the dataset into train, validation, and test sets as described in the paper.

After downloading the zip files pointed by the table links, you will have access to the following data:

- train, validation, and test sets
- a file to map visual features to items (to be removed when the code is integrated within the new version of Elliot)
- the mapping between users and items ids and their original Amazon unique codes
- the extracted visual features to train and evaluate the baselines and the proposed model

If you do not want to modify the configuration files, then you need to create a folder ```./data/``` where you will place the two downloaded datasets. Otherwise, feel free to change the paths from the configuration files and use your own setting.

**Disclaimer.** As we do not own the dataset, we decided not to release the original product images (you would need them to train DVBPR, or to extract each of the visual features - we already extracted them for you, though). Once again, we encourage you refer to the [official link](https://jmcauley.ucsd.edu/data/amazon/) to get to the full dataset, and use the item mapping we provided to download again the original product images, if needed.

**Visual Features.** All visual features were extracted using the scripts accessible at [this repository](https://github.com/sisinflab/Multimodal-Feature-Extractor). The repository is maintained by our research group, and its purpose is to provide a comprehensive framework to extract both handcrafted and trainable features from multimedia data, e.g., images and texts. Please, feel free to join us and contribute to it, since its developing is still open and highly active!

## Baselines and Our Method

### Traditional Collaborative Filtering
|       Model      |    Paper  |
| ------------------ | ------------------ |
|     BPRMF     | [Rendle et al.](https://arxiv.org/pdf/1205.2618.pdf) |
|     NeuMF   | [He et al.](https://arxiv.org/pdf/1708.05031.pdf) |


### Visual-Based Collaborative Filtering
|       Model      |    Paper  |
| ------------------ | ------------------ |
|     VBPR     | [He and McAuley](https://arxiv.org/pdf/1510.01784.pdf) |
|     DeepStyle   | [Liu et al.](http://www.shuwu.name/sw/DeepStyle.pdf) |
|     DVBPR   | [Kang et al.](https://arxiv.org/pdf/1711.02231.pdf) |
|     ACF | [Chen et al.](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-AttentiveCF.pdf) | 
|     VNPR | [Niu et al.](https://people.engr.tamu.edu/caverlee/pubs/niu18wsdm.pdf) |

### Our Model
You may find the scripts for our proposed method [here](https://github.com/sisinflab/Content-Style-VRSs/tree/master/elliot/recommender/proposed). We renamed our model CSV, which stands for <ins>**C**</ins>ontent-<ins>**S**</ins>tyle <ins>**V**</ins>isual Recommendation.

## Reproducibility Details

We randomly initialize the model parameters of all tested methods with a Gaussian distribution with a mean of 0 and standard deviation of 0.01 and set the latent factor dimension to 128 following the experimental settings used in [Chen et al.](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Attentive%20Collaborative%20Filtering%20Multimedia%20Recommendation%20with%20Item-%20and%20Component-Level%20Attention.pdf). We explore the following hyperparameters via grid-search: the learning rate in {0.0001, 0.001, 0.01} and the regularization coefficients in {0.00001, 0.001}, whereas we fix the batch size to 256, and the temperature to 10.0. We adopt early-stopping to avoid overfitting and choose the best model configuration for each algorithm according to the hit ratio (HR), i.e., the validation metric, as in [Chen et al.](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Attentive%20Collaborative%20Filtering%20Multimedia%20Recommendation%20with%20Item-%20and%20Component-Level%20Attention.pdf).

That being said, you may refer to the two configuration files:
- [Amazon Boys & Girls](https://github.com/sisinflab/Content-Style-VRSs/blob/master/config_files/evaluate_amazon_boys_girls.yml)
- [Amazon Men](https://github.com/sisinflab/Content-Style-VRSs/blob/master/config_files/evaluate_amazon_men.yml)

to have a detailed overview on all adopted hyperparameters, evaluation settings, and files to run the experiments. The files are in yml format (as required in Elliot), so their meaning should be easy to follow and understand.

## Contacts
* Yashar Deldjoo (yashar.deldjoo@poliba.it)
* Tommaso Di Noia (tommaso.dinoia@poliba.it)
* Daniele Malitesta* (daniele.malitesta@poliba.it)
* Felice Antonio Merra (felmerra@amazon.de)

\**corresponding author*
