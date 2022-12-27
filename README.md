# A Novel Federated Learning Scheme for Generative Adversarial Networks

Collaborative Game Parallel Learning

A distributed learning algorithm for GANs with Non-IID dataset

## Env-requirements

- torch
- matplotlib
- fedlab

# Comparison Algorithm
- AC-GAN
- FL-GAN
- MD-GAN
- FeGAN
- CAP-GAN (ours)



## Usage
All codes use global variants to control the experiment setting 
Regular parameters
```angular2html
num_workers :   The number of clients in the system
num_servers :   The number of edge servers in the system
E : To control when to share the discriminator to neighbors. Note that: We provide the code in notes and just cancel the note if you want to test it
num_class :     This parameter only acts on generating how many classes of 2D-Gaussian Mixture 
num_sample : To control the number of samples for testing
batch_size : Batch Size
frac_workers: It is for FL-GAN or FeGAN
epoch:  The epoch for local iterations in clients.
```

### ACGAN
Enter /ACGAN/2DMG (MNIST)/

``
python acgan.py
``


### MDGAN
Enter /MDGAN/2DMG (MNIST)/

``
python mdgan.py
``


### FeGAN

``
python fegan.py
``

### CAPGAN
Enter /CAPGAN/MNIST (MNIST)/

``
python main.py
``

Enable Mix-G module (mixgan)
Enter the /CAPGAN/MNIST/

``
python mixed-gan.py
``

