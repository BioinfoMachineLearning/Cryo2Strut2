# Cryo2Struct2


Cryo2Struct2 is a fully automated method for modeling 3D atomic structures from cryo-EM density maps, building on its predecessor, Cryo2Struct. It employs a multi-task deep learning model that integrates sequence-based features from a Protein Language Model (ESM) with cryo-EM density maps, merging feature representation across modalities. The predicted voxels are then used to construct a Hidden Markov Model (HMM), followed by a customized Viterbi algorithm to align sequences and generate initial protein backbone structures. These backbone models are used as templates for AlphaFold3, which further refines the structures for improved accuracy. By integrating cryo-EM data with AlphaFold3 predictions, Cryo2Struct2 improves structure refinement and helps AlphaFold3 to predict accurate structures.

## Setup Environment (Locally)
To setup Cryo2Struct2 locally, follow the steps below. It takes about 3-7 minutes to set up the environment to run Cryo2Struct2.

Clone this repository and `cd` into it
```
git clone https://github.com/BioinfoMachineLearning/Cryo2Strut2.git
cd ./Cryo2Struct2
```

We will set up the environment using Anaconda. This is an example of setting up a conda environment to run the code. Use the following command to create the conda environment using the ``cryo2struct2.yml`` file.

```
conda env create -f cryo2struct2.yml
conda activate cryo2struct2
```

## Atomic structure modeling using Cryo2Struct2

1. <ins>**Input**</ins>: **cryo-EM density map and sequence** : First, you need to prepare your own data or use our provided example data. The directory should be organized as follows:
```text 
cryo2struct
|── input
    │── 34610
        │-- emd_34610.map
        |-- 8hb0.fasta
        |-- 8hb0.pdb
```
The `emd_34610.map` is the density map with EMD ID: 34610 downloaded from EMDB website. The `8hb0.fasta` is the corresponding sequence file.  

The `8hb0.pdb` file is a PDB structure file used in this test example to generate embeddings using ESM. Alternatively, users can use the `8hb0.fasta` file to generate embeddings from ESM.

The first step is to make input cryo-EM map ready for Cryo2Struct. We run [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/index.html) in non-GUI mode to resample the density map to 1 Angstrom, please install it to preprocess the map. We used ChimeraX 1.4-1 in CentOS 8 system. Once ChimeraX is installed, then please run the following.

```
bash preprocess/run_data_preparation.bash input/
```
In the above example ``input/`` is the ``absolute input path`` where the maps are present.

**Note**: For this example, the normalized map is provided, so there is no need to run the above bash command to prepare the map. Hence, the directory structure for this example looks like this:

```text 
cryo2struct
|── input
    │── 34610
        │-- emd_34610.map
        |-- emd_normalized_map.mrc
        |-- 8hb0.fasta
        |-- 8hb0.pdb
```
2. <ins>**Set Up ESM**</ins>:
Set up ESM in your system following the instruction provided in https://github.com/facebookresearch/esm . The esm.pretrained model we used is `esm2_t36_3B_UR50D()`. Change the path of saved ESM model in [utils/grid_division.py](utils/grid_division.py).

3. <ins>**Running Cryo2Struct2**</ins>:
The deep learning requires trained atom and amino acid type models. The trained models are available in [Cryo2Struct2 Harvard Dataverse](https://doi.org/10.7910/DVN/YYHWZO). Use the following to download the trained models. 

```
cd models
wget -O amino_acid_type.ckpt https://dataverse.harvard.edu/api/access/datafile/10888677
wget -O atom_type.ckpt https://dataverse.harvard.edu/api/access/datafile/10888678
cd ..
```

The organization of the downloaded models should look like:
```text 
cryo2struct
|── input
    │── 34610
        │-- emd_34610.map
        |-- emd_normalized_map.mrc
        |-- 8hb0.fasta
        |-- 8hb0.pdb
|── models
    │-- amino_acid_type.ckpt
    |-- atom_type.ckpt
    |-- aa_regression_model.pkl
    |-- ca_regression_model.pkl
```

Update the configurations in the [config/arguments.yml](config/arguments.yml) file. Especialy the input data directory, trained model checkpoint path,  and density map name. By default the program runs inference in `CPU`, running the inference program on the ``GPU`` speeds up prediction. To enable ``GPU`` processing, modify ``infer_run_on`` in the configuration file to ``gpu`` and provide the GPU device id on ``infer_on_gpu`` (example: 0). One way to update the configuration by using visual editor (``vi``).

```
vi config/arguments.yml
```

<ins>**Compile Modified Viterbi algorithm:**</ins>
The Hidden Markov Model-guided carbon-alpha alignment programs are available in [viterbi/](viterbi/). The alignment algorithm is written in C++ program, so compile them using: 

```
cd viterbi
g++ -fPIC -shared -o viterbi.so viterbi.cpp -O3
cd ..
```
During the compilation, if the program asks for installation of `gcc-c++` package, then install it following the instructions. GCC C++ compiler is required to compile `viterbi.cpp`.

If the compilation of the program fails due to library issues (which typically occurs when attempting to compile on older systems), you can try compiling using the following approach:
```
cd viterbi
conda install -c conda-forge gxx
g++ -fPIC -shared -o viterbi.so viterbi.cpp -O3
cd ..
```
The above command installs the ``gxx`` package in the activated conda environment, which provides the GCC C++ compiler. This compiler is useful for compiling C++ code on the system. The HMM alignment program runs on the ``CPU`` and is optimized at the highest level using the``-O3`` flag. We tested, and the above compilation was successful on CentOS 7, 8, and AlmaLinux OS 8.8, 8.9. 

Finally, run the following:

```
python3 cryo2struct2.py --density_map_name 34610
```

4. <ins>**Output**</ins>:  **Modeled atomic structure**
The output model will be saved in the density map's directory. 


5. <ins>**Integrating Cryo2Struct2 Models as Templates for AlphaFold3**</ins>: 
The models generated by Cryo2Struct2 are used as templates for AlphaFold3. Use the provided script [prepare_script_af3_multichain_multi_template.py](prepare_script_af3_multichain_multi_template.py) to generate `.json` files that will be used as input to run AlphaFold3.



6. <ins>**Set up AlphaFold3**</ins>: 
Request AlphaFold3 parameters and follow the instructions to set up AlphaFold3 from here : https://github.com/google-deepmind/alphafold3 .


7. <ins>**Run AlphaFold3**</ins>: 
Use the script [run_af3_docker_all.py](run_af3_docker_all.py) to run AlphaFold3 and to predict structures.


## Training Cryo2Struct Deep Learning
The training programs are available in the [train/](train/) directory. Cryo2Struct2 was trained on Cryo2StructData, which is accessible on the [Cryo2StructData Dataverse](https://doi.org/10.7910/DVN/FCDG0W). Download the full dataset from [Cryo2Struct Full Dataset](https://doi.org/10.7910/DVN/FCDG0W) or a small subset from [Cryo2Struct Small Subsample Dataset](https://doi.org/10.7910/DVN/CGUENL). After downloading the dataset, `unzip` the compressed files. The directory names are the EMD ID of the cryo-EM density map.

The dataset contains the preprocessed map ready for deep learning training. However, the cryo-EM density map label needs to be prepared. Run the following


```
python3 label/get_atoms_label.py density_map_directory
python3 label/get_amino_labels.py density_map_directory
```

The `density_map_directory` is the absolute directory path where unzipped cryo-EM density maps are present. The above scripts generate the atom and amino acid-type labels, which are used during the training of the deep learning model.

Split the data into training and validation sets. If you choose to use our predefined training and validation splits, refer to the Excel sheet in [Cryo2StructData Metadata](https://doi.org/10.7910/DVN/JMN60H), which contains the IDs for the training and validation cryo-EM density maps. Create separate directories for training and validation, and move the corresponding data to each directory.


Generate sub-grids of cryo-EM density maps from training and validation dataset for training. These sub-grids are used for training the model. Run the following:

```
python3 train/grid_division_train.py train_map_directory train_sub_grids
python3 train/grid_division_train.py valid_map_directory valid_sub_grids
```

The `train_map_directory` is the directory containing training cryo-EM density maps, and `train_sub_grids` is the directory where the training sub-grids will be generated. Similarly, `valid_map_directory` is the directory containing validation cryo-EM density maps, and `valid_sub_grids` is the directory where the validation sub-grids will be generated. After generation of sub-grids, run:

```
ls train_sub_grids > train_splits.txt
ls valid_sub_grids > valid_splits.txt
```

We used the distributed data parallel (DDP) technique to train the models on 24 compute nodes, each equipped with 6 NVIDIA V100 GPUs with 32GB of memory. The training program can run on a single GPU, multiple GPUs, or a multi-node cluster with multiple GPUs. Finally, in the training scripts [train/train.py](train/train.py) change the values in `AVAIL_GPUS` to the number of GPUs available in the compute node, `NUM_NODES` to the number of available compute nodes, and set `BATCH_SIZE`, and `DATASET_DIR` to the path of the Cryo2Struct directory. Then, train the model by running:

```
python3 train/train.py    # trains both amino acid-type and atom prediction model
```
Monitor the training progress in [Weights and Biases](https://wandb.ai/site).


Optional: The source code for data preprocessing, label generation and validation of training data is available at [Cryo2StructData GitHub repository](https://github.com/BioinfoMachineLearning/cryo2struct).


## Contact Information
If you have any question, feel free to open an issue or reach out to us: [ngzvh@missouri.edu](ngzvh@missouri.edu), [chengji@missouri.edu](chengji@missouri.edu).

## Acknowledgements
We thank the High-Performance Computing (HPC) resource, Hellbender, located at the University of Missouri, Columbia, MO, which was used for training, inference and alignment process.

