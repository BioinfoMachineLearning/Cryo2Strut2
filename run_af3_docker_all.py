import os
import subprocess

# Directory containing your JSON files. These are generated by prepare_script_af3_multichain_multi_template.py program.
input_directory = "/af_scripts" 
count = 0
# Iterate through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):  # Ensure only JSON files are processed
        json_path = f"/root/af_inputs_multichain_multitemplate/{filename}"
        
        # Construct the Docker command
        docker_command = [
            "docker", "run", "-it",
            "--volume", "/AF3/alphafold3/af_inputs_multichain_multitemplate:/root/af_inputs_multichain_multitemplate",
            "--volume", "/AF3/alphafold3/af_outputs_multichain_multitemplate:/root/af_outputs_multichain_multitemplate",
            "--volume", "/AF3/alphafold3/model_parameters:/root/models",
            "--volume", "/databases:/root/public_databases",
            "--gpus", "all",
            "alphafold3",
            "python", "run_alphafold.py",
            f"--json_path={json_path}",
            "--model_dir=/root/models",
            "--output_dir=/root/af_outputs_multichain_multitemplate"
        ]
        
        print(f"Running for file: {filename}")
        
        # Run the Docker command
        print(docker_command)
       
        subprocess.run(docker_command)
        
        print(f"Completed for file: {filename}")
        count += 1
        print("Completed total: ", count)
