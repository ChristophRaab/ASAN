# Adversarial Spectral Adaptation Network (ASAN).
Pytorch Source code for **Adversarial Spectral Adaptation Network (ASAN)**.<br>
This repository is part of the ACCV 2020 submission **Bridging Adversarial and Statistical DomainTransfer via Spectral Adaptation Networks**. <br>
[Link to paper](https://openaccess.thecvf.com/content/ACCV2020/html/Raab_Bridging_Adversarial_and_Statistical_Domain_Transfer_via_Spectral_Adaptation_Networks_ACCV_2020_paper.html) <br>
Contact: christoph.raab@fhws.de

## Installation
1. We provide the yml-file for the virtual conda envirement in the file <br>
   [pytorch.yml](https://github.com/sub-dawg/ASAN/blob/master/pytorch.yml).   After installation you should be able to run the code in this repository.

2. First, download the dataset your are prefering.
   The links to the datasets can be found in the respective sections.
3. After download, unzip the files and place the output folders as they are in the directory [images](https://github.com/sub-dawg/ASAN/blob/master/images/).
4. (Optional: Adapt the dataset lists in [data](https://github.com/sub-dawg/ASAN/blob/master/data/) to your to preferred path)

## Demo
For a simple training-evaluation demo run with preset parameters, you can use the following commands for training on

**Office-31 A->W**<br>
`train_image.py --tl RSL --s_dset_path data/office/amazon.txt --t_dset_path data/office/webcam.txt --test_interval 100 --num_workers 12 --sn True --k 11 --tllr 0.001`

**Image-Clef I->P**<br>
`train_image.py --tl RSL --dset image-clef --s_dset_path data/office/amazon.txt --t_dset_path data/office/webcam.txt --test_interval 100 --num_workers 12 --sn True --k 11 --tllr 0.001`

## Training and Inference
1. The network can be trained via train_image.py
   See the Args-Parser parameter description in the file for the documentation of the parameters.

2. The trained models is obtainable under snapshopt/san


## Datasets
### Office-31
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).

### Office-Home
Office-Home dataset can be found [here](https://drive.google.com/file/d/1W_U8GsILKdMSxqhnmTbYaaWhvQ-P4RJ1/view?usp=sharing).

### Image-clef
Image-Clef dataset can be found [here](https://drive.google.com/file/d/1lu1ouDoeucW8MgmaKVATwNt5JT7Uvk62/view?usp=sharing).


## Acknowledgement
We thank Mingsheng Long et al. for providing the code for the CDAN model.<br>
See https://github.com/thuml/CDAN

