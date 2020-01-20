# Collaborative Sampling for Image Inpainting

## Authors 

- Thevie Mortiniera
- Yuejiang Liu (Supervisor)
- Alexandre Alahi (Professor)


## Inpainting

Results on FASHION_MNIST

<p float="left">
  <img src="../master/assets/inpaint1.png" width="32%"> 
  <img src="../master/assets/inpaint2.png" width="32%"> 
  <img src="../master/assets/inpaint4.png" width="32%">
</p>

Sampling results for image inpainting by targeting the corrupted region. (Top) Input data with masked region(second row) Output of the generator surrounded by itscontext image (third row) Heatmap highlighting visualdifferences between the inpainted output of thegenerator in the 2nd row and the refined results in thefourth row. The closer to the red, the higher thedifferences (fourth row) refined samples after applyingcollaborative sampling and discriminator shapingsurrounded by its context image (bottom) Originalimages.

## Documentation

### Download dataset

The following command allow to download the FASHION-MNIST data set or CelebA and create the corresponding folders as in the directory hierarchy below.

``` python download.py [fashion_mnist | celebA] ```

### Directory hierarchy

```
.
│   collaborative-image-inpainting
│   ├── src
│   │   ├── collaborator.py
│   │   ├── dataset.py
│   │   ├── dcgan.py
│   │   ├── download.py
│   │   ├── inpaint_main.py
│   │   ├── inpaint_model.py
│   │   ├── inpaint_solver.py
│   │   ├── main.py
│   │   ├── mask_generator.py
│   │   ├── ops.py
│   │   ├── policy.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   │   └── utils_2.py
│   Data
│   ├── celebA
│   │   ├── train
│   │   └── val
│   ├── fashion_mnist
│   │   ├── train
│   │   └── val

```
