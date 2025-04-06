#!/usr/bin/env python3
"""
RAVE Latent Space Analysis to Dimensionality Reduced Coordinates
Consolidated version that supports exporting either a 2D or 3D mapping (or any n-dimensional mapping)
from the RAVE latent space using PCA, T-SNE, or UMAP.

Original author: Moisés Horta Valenzuela
Modified by: David Piazza
Last Modified: 06/04/2025
"""

import argparse
import torch
import librosa as li
import numpy as np
from tqdm import tqdm
import os
import json

# For PCA and T-SNE:
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# For UMAP (make sure to install: pip install umap-learn)
from umap import UMAP

# For OSC functionality (make sure to install: pip install python-osc)
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import threading
import time

def process_audio_files(model_path, audio_dir, output_json=None, n_components=2, sr=48000, method="pca", skip_dim_reduction=False, osc_client=None, osc_address=None):
    """Process audio files using the RAVE model and dimensionality reduction."""
    # Set device: GPU if available, otherwise CPU.
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained RAVE model.
    print(f"Loading RAVE model from {model_path}...")
    rave = torch.jit.load(model_path).to(device)
    rave.eval()

    # List all .wav files in the specified audio directory.
    audio_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith('.wav')
    ]
    if not audio_files:
        print(f"No audio files found in directory: {audio_dir}")
        return None

    # Dictionary to store latent vectors for each file
    latent_data = {}
    latent_vectors_all = []
    num_dimensions = None

    print("Encoding audio files to latent space...")
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Get filename without extension for use as key
        file_key = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Load audio file using librosa.
        x, _ = li.load(audio_file, sr=sr, mono=True)
        # Reshape to (1, 1, samples) and move to the device.
        x = torch.from_numpy(x).reshape(1, 1, -1).float().to(device)

        # Encode the audio into latent representation using the RAVE model.
        with torch.no_grad():
            z = rave.encode(x)  # Expected shape: (1, n_dimensions, encoded_sample_length)
            z = z.cpu().numpy().squeeze()  # Remove batch dimension: (n_dimensions, encoded_sample_length)
            
            # Check dimension of z and handle accordingly
            if z.ndim == 1:
                # Only one dimension, so we have a single vector, no need to average
                z_mean = z
                # If this is the first file, use z's length as num_dimensions
                if num_dimensions is None:
                    num_dimensions = 1
            else:
                # Store the number of dimensions (for the 'cols' field in output JSON)
                if num_dimensions is None:
                    num_dimensions = z.shape[0]
                
                # Calculate the mean latent vector for the file (averaging across time)
                z_mean = np.mean(z, axis=1)
            
            # Store the mean latent vector in the dictionary with filename as key
            latent_data[file_key] = z_mean.tolist()
            latent_vectors_all.append(z_mean)

    # Prepare output data in the format expected by fluid.dataset~
    output_data = {
        "cols": num_dimensions,
        "data": latent_data
    }

    # Optionally perform dimensionality reduction and include it in the output
    if not skip_dim_reduction:
        latent_matrix = np.array(latent_vectors_all)
        
        # Always use 2 components for dimensionality reduction regardless of user input
        n_components = 2
        
        if method == "tsne":
            print("Performing T-SNE on latent vectors (2 components)...")
            tsne = TSNE(n_components=n_components, random_state=42)
            latent_reduced = tsne.fit_transform(latent_matrix)
            # Scale T-SNE coordinates
            max_val = np.abs(latent_reduced).max()
            if max_val != 0:
                latent_reduced = latent_reduced * (5.0 / max_val)
            output_data["reduced_data"] = {}
            for i, file_key in enumerate(latent_data.keys()):
                output_data["reduced_data"][file_key] = latent_reduced[i].tolist()
        
        elif method == "umap":
            print("Performing UMAP on latent vectors (2 components)...")
            umap_model = UMAP(n_components=n_components, random_state=42)
            latent_reduced = umap_model.fit_transform(latent_matrix)
            # Scale UMAP coordinates
            max_val = np.abs(latent_reduced).max()
            if max_val != 0:
                latent_reduced = latent_reduced * (5.0 / max_val)
            output_data["reduced_data"] = {}
            for i, file_key in enumerate(latent_data.keys()):
                output_data["reduced_data"][file_key] = latent_reduced[i].tolist()
                
        else:  # pca
            print("Performing PCA on latent vectors (2 components)...")
            pca = PCA(n_components=n_components)
            latent_reduced = pca.fit_transform(latent_matrix)
            output_data["reduced_data"] = {}
            for i, file_key in enumerate(latent_data.keys()):
                output_data["reduced_data"][file_key] = latent_reduced[i].tolist()

    # Determine the output JSON filename.
    if output_json is None:
        base_name = os.path.basename(model_path).split('.')[0]
        if skip_dim_reduction:
            output_json = f"latent_vectors_{base_name}.json"
        else:
            output_json = f"2D_{method}_latent_mapping_{base_name}.json"
    else:
        # Ensure the output file has a .json extension
        if not output_json.lower().endswith('.json'):
            output_json += '.json'
    
    # Convert to absolute path if not already absolute
    if not os.path.isabs(output_json):
        output_json = os.path.abspath(output_json)

    # Save the output data to a JSON file.
    with open(output_json, "w") as json_file:
        json.dump(output_data, json_file, indent=2)

    print(f"Latent mapping data saved to {output_json}")
    
    # Send OSC message with the path to the output file if client is provided
    if osc_client and osc_address:
        osc_client.send_message(osc_address, output_json)
        print(f"Sent OSC message to {osc_address} with output path: {output_json}")
    
    return output_json

def handle_process_request(address, *args):
    """Handle OSC message with audio directory and model path."""
    if len(args) < 2:
        print(f"Not enough arguments received at {address}. Expected audio_dir and model_path.")
        return
    
    # Get path arguments and normalize them
    audio_dir = args[0]
    model_path = args[1]
    
    # Fix macOS paths that might start with "Macintosh HD:"
    if audio_dir.startswith("Macintosh HD:"):
        audio_dir = audio_dir.replace("Macintosh HD:", "", 1)
    if model_path.startswith("Macintosh HD:"):
        model_path = model_path.replace("Macintosh HD:", "", 1)
    
    # Optional arguments
    output_json = args[2] if len(args) > 2 else None
    method = args[3] if len(args) > 3 else "pca"
    skip_dim_reduction = bool(args[4]) if len(args) > 4 else False
    
    print(f"Processing audio files in {audio_dir} with model {model_path}")
    
    # Process in a separate thread to not block the OSC server
    def process_thread():
        output_path = process_audio_files(
            model_path=model_path,
            audio_dir=audio_dir,
            output_json=output_json,
            method=method,
            skip_dim_reduction=skip_dim_reduction,
            osc_client=osc_client,
            osc_address="/rave/processing/done"
        )
    
    threading.Thread(target=process_thread).start()

def get_model_dimensions(model_path, osc_client=None, osc_address=None):
    """Load a RAVE model and determine its latent dimensions."""
    try:
        # Fix macOS paths that might start with "Macintosh HD:"
        if model_path.startswith("Macintosh HD:"):
            model_path = model_path.replace("Macintosh HD:", "", 1)
            
        print(f"Loading model to determine latent dimensions: {model_path}")
        
        # Set device: GPU if available, otherwise CPU
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        rave = torch.jit.load(model_path).to(device)
        rave.eval()
        
        # Create a short test signal to encode
        test_signal = torch.zeros(1, 1, 48000).float().to(device)  # 1 second of silence at 48kHz
        
        # Encode to get dimensions
        with torch.no_grad():
            z = rave.encode(test_signal)
            latent_dim = z.shape[1]  # The latent dimension is the second dimension
        
        print(f"Model has {latent_dim} latent dimensions")
        
        # Send OSC message with the dimensions if client is provided
        if osc_client and osc_address:
            osc_client.send_message(osc_address, latent_dim)
            print(f"Sent OSC message to {osc_address} with dimensions: {latent_dim}")
        
        return latent_dim
        
    except Exception as e:
        print(f"Error determining model dimensions: {str(e)}")
        if osc_client and osc_address:
            osc_client.send_message(osc_address, -1)  # Send error code
        return None

def handle_model_info_request(address, *args):
    """Handle OSC message requesting information about a model."""
    if not args:
        print(f"No model path received at {address}")
        return
    
    model_path = args[0]
    print(f"Received request for model info: {model_path}")
    
    # Process in a separate thread to not block the OSC server
    def process_thread():
        get_model_dimensions(
            model_path=model_path,
            osc_client=osc_client,
            osc_address="/rave/model/dimensions"
        )
    
    threading.Thread(target=process_thread).start()

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="RAVE Latent Space Analysis with OSC Server"
    )
    parser.add_argument(
        "--osc_ip",
        type=str,
        default="127.0.0.1",
        help="IP address to listen for OSC messages (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--osc_in_port",
        type=int,
        default=9001,
        help="Port to listen for OSC messages (default: 9001)"
    )
    parser.add_argument(
        "--osc_out_port",
        type=int,
        default=9002,
        help="Port to send OSC messages (default: 9002)"
    )
    args = parser.parse_args()

    # Set up OSC dispatcher
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/rave/process", handle_process_request)
    osc_dispatcher.map("/rave/model/info", handle_model_info_request)
    
    # Create OSC client for sending responses
    global osc_client
    osc_client = udp_client.SimpleUDPClient(args.osc_ip, args.osc_out_port)
    
    # Start OSC server
    server = osc_server.ThreadingOSCUDPServer(
        (args.osc_ip, args.osc_in_port), osc_dispatcher)
    
    print(f"Starting OSC server at {args.osc_ip}:{args.osc_in_port}")
    print(f"Send messages to /rave/process with arguments: [audio_dir] [model_path] [optional: output_json] [optional: method] [optional: skip_dim_reduction]")
    print(f"Send messages to /rave/model/info with argument: [model_path] to get latent dimensions")
    print(f"Responses will be sent to {args.osc_ip}:{args.osc_out_port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOSC server stopped.")

if __name__ == "__main__":
    main()