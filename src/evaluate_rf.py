import os
import joblib
import cityflow
import pandas as pd

def evaluate_pure_rf(config_path="data/config.json", model_path="models/rf_pretrain.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run pretrainer.py first!")
        
    eng = cityflow.Engine(config_path, thread_num=4)
    rf_model = joblib.load(model_path)
    expected_features = getattr(rf_model, "feature_names_in_", None)
    
    print("Evaluating Pure Pre-Training Policy (Before Q-Learning)...")
    step = 0
    metrics = []
    
    while step < 3600:
        eng.next_step()
        lane_vehicle_counts = eng.get_lane_vehicle_count()
        
        # Build feature vector matching training columns
        state_df = pd.DataFrame([lane_vehicle_counts])
        if expected_features is not None:
            for col in expected_features:
                if col not in state_df.columns:
                    state_df[col] = 0
            state_df = state_df[expected_features]
            
        # Predict action using pure Random Forest heuristic policy
        action = int(rf_model.predict(state_df)[0])
        
        # Log metrics
        total_vehicles = sum(lane_vehicle_counts.values())
        avg_travel_time = eng.get_average_travel_time()
        
        metrics.append({
            "step": step,
            "total_queue": total_vehicles,
            "avg_travel_time": avg_travel_time
        })
        step += 1
        
    df_metrics = pd.DataFrame(metrics)
    output_report = "data/rf_baseline_metrics.csv"
    df_metrics.to_csv(output_report, index=False)
    
    print(f"Pure RF Evaluation Complete! Metrics saved to {output_report}")
    print(f"Baseline Average Travel Time: {df_metrics['avg_travel_time'].iloc[-1]:.2f}s")
    print(f"Baseline Peak Queue Length: {df_metrics['total_queue'].max()} vehicles")

if __name__ == "__main__":
    evaluate_pure_rf()