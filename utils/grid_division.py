import numpy as np
import mrcfile
import os
import math
from copy import deepcopy
import torch
import esm
from pathlib import Path

box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundry issues



os.environ["TORCH_HOME"] = "/esm"

def chain_merger_2(density_map, fasta_name):

    input_file = f'{density_map}/atomic.fasta'

    output_file = f'{density_map}/{fasta_name}_all_chain_combined.fasta'

    # if os.path.exists(output_file):
        # os.remove(output_file)

    with open(input_file, "r") as input_fp, open(output_file, "w") as output_fp:
        merge_lines = []
        repeat_count = 1

        for line in input_fp:
            if line.startswith(">"):
                if merge_lines:
                    merged_line = "".join(merge_lines) * repeat_count
                    output_fp.write(merged_line)
                    merge_lines = []
                    repeat_count = 1

                chain_info = line.split("|")
                if len(chain_info) > 1:
                    chain_data = chain_info[1]
                    repeat_count = len(chain_data.split(","))
            else:
                merge_lines.append(line.strip())

        if merge_lines:
            merged_line = "".join(merge_lines) * repeat_count
            output_fp.write(merged_line)



def generate_esm_embeddings(sequence, save_path):

    # Load the model and save it to the specified directory
    # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    with open(save_path, 'w') as file:
        for seq_res in sequence_representations[0]:
            file.write(str(float(seq_res)) + '\n')




def create_manifest(full_image):
    # creates a list of box_size tensors. Each tensor is passed to Transformer Unet independently
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size))
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = full_image
    manifest = list()

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        manifest.append(next_chunk)
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point  # Reset
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset
    return manifest


def get_data(density_map_dir):
    protein_manifest = None
    amino_manifest = None
    atom_manifest = None
    processed_maps = [m for m in os.listdir(density_map_dir)]
    for maps in range(len(processed_maps)):
        os.chdir(density_map_dir)
        if processed_maps[maps] == "emd_normalized_map.mrc":
            p_map = mrcfile.open(processed_maps[maps], mode='r')
            protein_data = deepcopy(p_map.data)
            protein_manifest = create_manifest(protein_data)

    return protein_manifest


def reconstruct_map(manifest, image_shape):
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimensions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2]))
    counter = 0
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]
                counter += 1
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image


def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions


def run_pdb2seq(pdb_file, perl_script_dir, atm_sequence):
    os.system("perl " + perl_script_dir + " " + pdb_file + ">>" + atm_sequence)


def create_subgrids(input_data_dir, density_map_name):
    density_map_dir = os.path.join(input_data_dir,density_map_name)
    pdb_files = [l for l in os.listdir(density_map_dir) if l.endswith(".fasta")]
    pdb_files.sort()
    pdb_name = pdb_files[0].split(".")[0]
    pdb_name = pdb_name.split("_")[0]
    pdb_name = pdb_name.lower()


    esm_embeddings = f"{density_map_dir}/atomic_esm_t36_3B_embeds.txt"

    if os.path.exists(esm_embeddings):
        os.remove(esm_embeddings)

    if not os.path.isfile(esm_embeddings):
        print("Generating Embeddings Using: ", pdb_name)
        
        atm_sequence = f'{density_map_dir}/atomic.fasta'

        if not os.path.isfile(atm_sequence):
            perl_script= "./preprocess/pdb2seq.pl"
            perl_script_expand = os.path.abspath(perl_script)
            print(perl_script_expand)
            run_pdb2seq(pdb_file=f"{density_map_dir}/{pdb_name}.pdb",perl_script_dir=perl_script_expand, atm_sequence=atm_sequence)
        
        with open(atm_sequence,'r') as all_fasta:
            combined_sequence = all_fasta.read()


        generate_esm_embeddings(sequence=combined_sequence, save_path=esm_embeddings)

    # esm_embeddings = f"{density_map_dir}/{pdb_name}_esm_t36_3B_embeds.txt"
    with open(esm_embeddings, 'r') as esm_emb:
        embeds = [float(line.strip()) for line in esm_emb.readlines()]


    protein = get_data(density_map_dir)
    if protein is not None:
        split_map_dir = os.path.join(density_map_dir, f"{density_map_name}_splits")
        os.makedirs(split_map_dir, exist_ok=True)
        for i in range(len(protein)):
            save_file_name = f'{split_map_dir}/{density_map_name}_{i}.npz'
            np.savez_compressed(file=save_file_name, protein_grid=protein[i], embeddings=embeds)
    else:
        print("There is no input map. Please check the input density map's directory")
        exit()



    # print("Done : ", density_map_name)

