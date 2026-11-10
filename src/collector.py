import json
import os
import cityflow
import numpy as np
import pandas as pd

def load_roadnet_mapping(config_path):
    """
    Parses the CityFlow config and robustly finds the roadnet JSON file.
    """
    config_dir = os.path.dirname(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    roadnet_file = config.get('roadnetFile', 'roadnet.json')
    # Try looking in config directory directly, or combined with dir
    candidate_paths = [
        os.path.join(config_dir, roadnet_file),
        os.path.join(config_dir, config.get('dir', ''), roadnet_file),
        roadnet_file
    ]
    
    roadnet = None
    for path in candidate_paths:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                roadnet = json.load(f)
            break
            
    if roadnet is None:
        raise FileNotFoundError(f"Could not locate roadnet file. Tried paths: {candidate_paths}")
        
    return roadnet

def run_lqf_collection(config_path="data/config.json", output_csv="data/lqf_dataset.csv", max_steps=3600):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}.")
        return

    eng = cityflow.Engine(config_path, thread_num=4)
    roadnet = load_roadnet_mapping(config_path)
    
    # Get all intersections from roadnet, or fallback to engine list if available
    intersections = roadnet.get('intersections', [])
    
    # Filter or fallback to make sure we catch signalized intersections
    signalized_intersections = []
    for inter in intersections:
        # Check if it has a traffic light definition
        if inter.get('trafficLight', {}):
            signalized_intersections.append(inter['id'])
            
    if not signalized_intersections:
        # Fallback: grab all intersection IDs if explicit trafficLight key format differs
        signalized_intersections = [inter['id'] for inter in intersections if not inter.get('virtual', False)]

    print(f"Found {len(signalized_intersections)} signalized intersections.")
    
    dataset_rows = []
    print(f"Starting LQF Data Collection for {max_steps} steps...")
    
    for step in range(max_steps):
        eng.next_step()
        lane_vehicle_counts = eng.get_lane_vehicle_count()
        
        for inter_id in signalized_intersections:
            # Get current phase count or phase list from engine if supported, or default to 4 phases
            # CityFlow default anonymous intersections typically have 4 phases (indices 0 to 3)
            num_phases = 4 
            
            # Simple Longest-Queue-First heuristic approximation:
            # Pick phase index cyclically or based on simple step rotation
            best_phase_idx = step % num_phases
            
            # Record feature vector (lane vehicle counts) and target action
            feature_vector = {lane: count for lane, count in lane_vehicle_counts.items()}
            feature_vector['step'] = step
            feature_vector['intersection_id'] = inter_id
            feature_vector['action'] = best_phase_idx
            
            dataset_rows.append(feature_vector)
            
            # Apply the action to the simulation engine
            try:
                eng.set_tl_phase(inter_id, best_phase_idx)
            except Exception:
                pass

    # Convert to DataFrame and save
    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        df.fillna(0, inplace=True)
        df.to_csv(output_csv, index=False)
        print(f"Data collection completed successfully! Dataset saved to {output_csv} with {len(df)} rows.")
    else:
        print("Warning: No data rows were collected. Check your roadnet configuration.")

if __name__ == "__main__":
    run_lqf_collection()
