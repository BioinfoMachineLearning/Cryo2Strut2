"""
author: nabin 

This script prepares input for running AlphaFold3.
1. Standard Input for AF3.
2. Input without using any templates.
3. Input using own template.


Runing AF3: 
    docker run -it \
    --volume /AF3/alphafold3/af_inputs:/root/af_input \
    --volume /AF3/alphafold3/af_outputs:/root/af_output \
    --volume /AF3/alphafold3/model_parameters:/root/models \
    --volume /bmlfast/databases:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/40352.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output 
    
    
Note: Replace [40352.json] with the input filename for AF3. This script prepares that.
"""

from Bio import SeqIO, pairwise2
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
import json
import os

from Bio.PDB import PDBParser, MMCIFIO
import os


no_temp_seq_list = list()

from Bio import PDB

restype_3to1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'UNK' : 'U',
}




def extract_seq(pdb_file, atomic_chain_seq_file, atomic_seq_file):
    chain_seq_dict = dict()
    parser = PDB.PDBParser()
    pdb_map = pdb_file
    struct = parser.get_structure("CA", pdb_map)
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        chain_id = chain.id
                        try:
                            amino_name = restype_3to1[residue.resname]
                            if chain_id in chain_seq_dict:
                                chain_seq_dict[chain_id].append(amino_name)
                            else:
                                chain_seq_dict[chain_id] = [amino_name]
                        except KeyError:
                            pass
                        
    with open(atomic_chain_seq_file, 'w') as a_c:
        for k,v in chain_seq_dict.items():
            print(f">pdb2seq|Chains {k}", file=a_c)
            result = ''.join(v)
            print(result, file=a_c)
 
    all_seq = list()
    with open(atomic_seq_file, 'w') as a_s:
        print(">pdb2seq|Chains A", file=a_s)
        for k,v in chain_seq_dict.items():
            result = ''.join(v)
            all_seq.append(result)
        final_result = ''.join(all_seq)

        print(final_result,file=a_s)


def extract_sequences_from_fasta(fasta_file):
    """
    Extract sequences and chain IDs from a multi-chain FASTA file.
    """
    sequences = {}
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    current_chain = None
    for line in lines:
        if line.startswith(">"):
            current_chain = line.split("|")[1].strip().split()[1]
            sequences[current_chain] = ""
        else:
            sequences[current_chain] += line.strip()
    
    return sequences

def extract_sequences_from_structure_multichain(structure_file):
    """
    Extract sequences from a multi-chain structure file (mmCIF or PDB).
    """
    if structure_file.endswith(".cif"):
        parser = MMCIFParser()
    elif structure_file.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Invalid file format. Use .cif or .pdb.")
    
    structure = parser.get_structure("template", structure_file)
    chain_sequences = {}

    for chain in structure.get_chains():
        chain_id = chain.id
        sequence = ""
        for residue in chain.get_residues():
            if residue.has_id("CA"):  # Only include residues with CA (amino acids)
                sequence += seq1(residue.get_resname(), custom_map={"UNK": "X"})
        chain_sequences[chain_id] = sequence

    return chain_sequences

def align_sequences(query_sequence, template_sequence):
    """
    Align the query sequence with the template sequence and get indices for AlphaFold3.
    """
    alignments = pairwise2.align.globalxx(query_sequence, template_sequence)
    best_alignment = alignments[0]
    
    query_indices = []
    template_indices = []
    query_pos = 0
    template_pos = 0
    
    for q, t in zip(best_alignment.seqA, best_alignment.seqB):
        if q != "-" and t != "-":
            query_indices.append(query_pos)
            template_indices.append(template_pos)
        if q != "-":
            query_pos += 1
        if t != "-":
            template_pos += 1
    
    return query_indices, template_indices

def align_sequences_multichain(query_sequences, template_sequences, protein_id):
    """
    Align multi-chain query sequences with template sequences and get indices for AlphaFold3.
    """
    alignment_data = {}

    # print("Query Sequences: ", query_sequences)
    # print("Template Sequences: ", template_sequences)

    for chain_id, query_sequence in query_sequences.items():
        if chain_id in template_sequences:
            template_sequence = template_sequences[chain_id]
            alignments = pairwise2.align.globalxx(query_sequence, template_sequence)
            best_alignment = alignments[0]
            
            query_indices = []
            template_indices = []
            query_pos = 0
            template_pos = 0
            
            for q, t in zip(best_alignment.seqA, best_alignment.seqB):
                if q != "-" and t != "-":
                    query_indices.append(query_pos)
                    template_indices.append(template_pos)
                if q != "-":
                    query_pos += 1
                if t != "-":
                    template_pos += 1

            alignment_data[chain_id] = (query_indices, template_indices)
        else:
            print(f"Warning: No template sequence found for chain {chain_id}. Skipping alignment.")
            alignment_data[chain_id] = ([], [])
            no_temp_seq_list.append(protein_id)
            
    return alignment_data

def pdb2cif(template_pdb, template_cif):

    # Load the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', template_pdb)

    # Write the initial mmCIF file
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(template_cif)

    release_date = "1999-01-01"  # change release date. this date is safe and AF3 includes it.
    
    # Need below for AF3 to find the release date. The cutoff is ~ 2021
    metadata_entries = f"""
    #
    _pdbx_database_status.recvd_initial_deposition_date {release_date}
    _struct_ref_seq.seq_release_date {release_date}
    _pdbx_audit_revision_history.revision_date {release_date}
    #
    """

    # Append the release date to the mmCIF file
    with open(template_cif, "a") as cif_file:
        cif_file.write(metadata_entries)

def extract_chain_from_structure(template_path, chain_id):
    """
    Extract the specified chain from a structure file and return its contents as a string.
    """
    from Bio.PDB import MMCIFParser, PDBIO, Select

    class ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return chain.id == self.chain_id

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", template_path)
    
    output_file = f"{os.path.splitext(template_path)[0]}_chain{chain_id}.cif"
    # io = PDBIO()
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file, select=ChainSelect(chain_id))

    release_date = "1999-01-01"  # change release date. this date is safe and AF3 includes it.
    
    # Need below for AF3 to find the release date. The cutoff is ~ 2021
    metadata_entries = f"""
    #
    _pdbx_database_status.recvd_initial_deposition_date {release_date}
    _struct_ref_seq.seq_release_date {release_date}
    _pdbx_audit_revision_history.revision_date {release_date}
    #
    """


    # Append the release date to the mmCIF file
    with open(output_file, "a") as cif_file:
        cif_file.write(metadata_entries)
    
    with open(output_file, "r") as file:
        chain_content = file.read()

    return chain_content

def generate_json_cryo2struct_multichain_2(query_sequences, alignment_data_1, alignment_data_2, template_path_1, template_path_2):
    """
    Generate JSON for multi-chain sequences with template mappings in AlphaFold3 format.
    Parameters:
        query_sequences: dict of {chain_id: query_sequence}
        alignment_data: dict of {chain_id: (query_indices, template_indices)}
        template_path: Path to the template structure file.
    Returns:
        dict: JSON structure for AlphaFold3 input.
    """
    name_ = os.path.basename(template_path_1).split("_")[0]
    name_ = f"{name_}_cryo2struct_multi_template"

    sequence_entries = []
    for chain_id, query_sequence in query_sequences.items():
        query_indices_1, template_indices_1 = alignment_data_1.get(chain_id, ([], []))

        query_indices_2, template_indices_2 = alignment_data_2.get(chain_id, ([], []))
        
        # Extract chain-specific template structure
        try:
            chain_template_content_1 = extract_chain_from_structure(template_path_1, chain_id)
            chain_template_content_2 = extract_chain_from_structure(template_path_2, chain_id)
        except Exception as e:
            print(f"Error extracting chain {chain_id} from template: {e}")
            chain_template_content_1 = ""
            chain_template_content_2 = ""

        sequence_entries.append({
            "protein": {
                "id": chain_id,
                "sequence": query_sequence,
                "pairedMsa": "",
                "unpairedMsa": "",
                "templates": [
                    {
                        "mmcif": chain_template_content_1.strip(),
                        "queryIndices": query_indices_1,
                        "templateIndices": template_indices_1
                    },
                    {
                        "mmcif": chain_template_content_2.strip(),
                        "queryIndices": query_indices_2,
                        "templateIndices": template_indices_2
                    }
                ]
            }
        })

    data = {
        "name": name_,
        "modelSeeds": [1],
        "sequences": sequence_entries,
        "dialect": "alphafold3",
        "version": 1
    }
    return data

def generate_json_cryo2struct(query_sequence, query_indices, template_indices, template_path):
    """
    Generate JSON with template mappings.
    """

    with open(template_path, "r") as cif_file:
        mmcif_contents = cif_file.read()
    mmcif_contents_inline = mmcif_contents.strip()


    name_ = os.path.basename(template_path).split("_")[0]
    name_ = f"{name_}_cryo2struct"
    data = {
        "name": name_,
        "modelSeeds": [1],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": query_sequence,
                    "pairedMsa": "",
                    "unpairedMsa":"",
                    "templates": [
                        {
                            "mmcif": mmcif_contents_inline,
                            "queryIndices": query_indices,
                            "templateIndices": template_indices
                        }
                    ]
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    return data


def generate_json_cryo2struct_multichain(query_sequences, alignment_data, template_path):
    """
    Generate JSON for multi-chain sequences with template mappings in AlphaFold3 format.
    """
    with open(template_path, "r") as cif_file:
        mmcif_contents = cif_file.read()
    mmcif_contents_inline = mmcif_contents.strip()

    name_ = os.path.basename(template_path).split("_")[0]
    name_ = f"{name_}_cryo2struct"

    sequence_entries = []
    for chain_id, query_sequence in query_sequences.items():
        query_indices, template_indices = alignment_data.get(chain_id, ([], []))
        sequence_entries.append({
            "protein": {
                "id": chain_id,
                "sequence": query_sequence,
                "pairedMsa": "",
                "unpairedMsa": "",
                "templates": [
                    {
                        "mmcif": mmcif_contents_inline,
                        "queryIndices": query_indices,
                        "templateIndices": template_indices
                    }
                ]
            }
        })

    data = {
        "name": name_,
        "modelSeeds": [1],
        "sequences": sequence_entries,
        "dialect": "alphafold3",
        "version": 1
    }
    return data



def generate_json_no_template(sequences, template_path):
    """
    Generate JSON for multi-chain sequences in AlphaFold3 format.
    """
    name_ = os.path.basename(template_path).split("_")[0]
    name_ = f"{name_}_no_template"

    sequence_entries = [
        {"protein": {"id": chain_id, "sequence": seq, "pairedMsa": "", "unpairedMsa": "", "templates": []}}
        for chain_id, seq in sequences.items()
    ]

    data = {
        "name": name_,
        "modelSeeds": [1],
        "sequences": sequence_entries,
        "dialect": "alphafold3",
        "version": 1
    }
    return data


def generate_json(sequences, template_path):
    """
    Generate JSON for multi-chain sequences in AlphaFold3 format.
    """
    name_ = os.path.basename(template_path).split("_")[0]
    name_ = f"{name_}_standard"

    sequence_entries = [
        {"protein": {"id": chain_id, "sequence": seq}}
        for chain_id, seq in sequences.items()
    ]

    data = {
        "name": name_,
        "modelSeeds": [1],
        "sequences": sequence_entries,
        "dialect": "alphafold3",
        "version": 1
    }
    return data


def main_multi(query_fasta_path, template_structure_1, template_structure_2, output_json_path, output_json_path_cryo2struct, output_json_path_no_template, protein_id):
    print("Extracting sequences from FASTA...")
    sequences = extract_sequences_from_fasta(query_fasta_path)
    print(f"Extracted sequences for chains: {', '.join(sequences.keys())}")

    print("Extracting template sequences from structure...")
    template_sequences_1 = extract_sequences_from_structure_multichain(template_structure_1)
    print(f"Extracted template sequences for chains: {', '.join(template_sequences_1.keys())}")

    template_sequences_2 = extract_sequences_from_structure_multichain(template_structure_2)
    print(f"Extracted template sequences for chains: {', '.join(template_sequences_2.keys())}")

    # print("Generating Standard JSON for multi-chain...")
    json_data = generate_json(sequences, template_structure_1)

    with open(output_json_path, "w") as outfile:
        json.dump(json_data, outfile, indent=2)
    print(f"JSON saved to {output_json_path}")

    print("Generating No Template JSON for multi-chain...")
    json_data = generate_json_no_template(sequences, template_structure_1)

    with open(output_json_path_no_template, "w") as outfile:
        json.dump(json_data, outfile, indent=2)
    print(f"JSON saved to {output_json_path_no_template}")

    alignment_data_1 = align_sequences_multichain(sequences, template_sequences_1, protein_id)
    alignment_data_2 = align_sequences_multichain(sequences, template_sequences_2, protein_id)

    print("Generating Cryo2Struct JSON for multi-chain...")
    json_data_cryo2struct = generate_json_cryo2struct_multichain_2(sequences, alignment_data_1, alignment_data_2, template_structure_1, template_structure_2)

    with open(output_json_path_cryo2struct, "w") as outfile:
        json.dump(json_data_cryo2struct, outfile, indent=2)
    print(f"JSON saved to {output_json_path_cryo2struct}")



if __name__ == "__main__":
    no_template_list = list()
    
    scripts_dir = "/AF3/alphafold3/af_inputs_multichain_multitemplate"

    os.makedirs(scripts_dir, exist_ok=True)


    data_dir = "/data"
    data_dir_list_all = os.listdir(data_dir)
    
    data_dir_list = data_dir_list_all


    for protein_id in data_dir_list:
        print("#"*50)
        print("Processing -> ", protein_id)
        print("#"*50)

        protein_dir = f"{data_dir}/{protein_id}"
        template_structure_pdb_1 = f"{protein_dir}/{protein_id}_cryo2struct_atomic_chain_conf_score_lmd.pdb"
        template_structure_1 = f"{protein_dir}/{protein_id}_cryo2struct_lmd.cif"

        template_structure_pdb_2 = f"{protein_dir}/{protein_id}_cryo2struct_atomic_chain_conf_score.pdb"
        template_structure_2 = f"{protein_dir}/{protein_id}_cryo2struct.cif"

        if os.path.exists(template_structure_pdb_2) and os.path.exists(template_structure_pdb_1):

            if os.path.exists(template_structure_1):
                os.remove(template_structure_1)
            
            if os.path.exists(template_structure_2):
                os.remove(template_structure_2)
            # convert pdb to cif. AF3 requires template in .cif format.
            pdb2cif(template_pdb=template_structure_pdb_1, template_cif=template_structure_1)
            pdb2cif(template_pdb=template_structure_pdb_2, template_cif=template_structure_2)

            fasta_file = [p for p in os.listdir(protein_dir) if p.endswith(".fasta")]
            fasta_file.sort()
            fasta_file_name = fasta_file[0].split(".")[0]
            protein_file = f"{protein_dir}/{fasta_file_name}.pdb"

            # print("Protein File: ", protein_file)

            atomic_chain_seq_file = f"{protein_dir}/atomic_seq_chain.fasta"
            atomic_seq_file = f"{protein_dir}/atomic_seq.fasta"

            if os.path.exists(atomic_chain_seq_file):
                os.remove(atomic_chain_seq_file)
            if os.path.exists(atomic_seq_file):
                os.remove(atomic_seq_file) 

        
            extract_seq(protein_file, atomic_chain_seq_file, atomic_seq_file)

            output_json_path = f"{scripts_dir}/{protein_id}.json"
            output_json_path_cryo2struct = f"{scripts_dir}/{protein_id}_cryo2struct_multi_template.json"
            output_json_path_no_template = f"{scripts_dir}/{protein_id}_no_template.json"

            if os.path.exists(output_json_path):
                os.remove(output_json_path)
                
            if os.path.exists(output_json_path_cryo2struct):
                os.remove(output_json_path_cryo2struct)

            if os.path.exists(output_json_path_no_template):
                os.remove(output_json_path_no_template)

            main_multi(atomic_chain_seq_file, template_structure_1, template_structure_2, output_json_path,  output_json_path_cryo2struct, output_json_path_no_template, protein_id)

            print("Done -> ", protein_id)
        else:
            print("No template structure found for ", protein_id)
            no_template_list.append(protein_id)

print("No template found for: ", no_template_list)
print("No template sequence found for: ", set(no_temp_seq_list))

print("Total Data: ", len(data_dir_list))
