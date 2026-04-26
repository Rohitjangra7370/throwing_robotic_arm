#!/bin/bash

# Target PID of the master training script
TARGET_PID=$1

if [ -z "$TARGET_PID" ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

echo "Monitoring PID $TARGET_PID (run_all_wind_experiments.py)..."
echo "Machine will automatically analyze results and shut down when finished."

# Wait until the PID is no longer active
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 60
done

echo "Training process finished at $(date)"

# Run the final analysis script and output to final_analysis.txt
echo "Running final analysis..."
python3 analyze_wind_results.py > final_analysis.txt
echo "Final analysis saved to final_analysis.txt."

# Shut down the system
echo "Shutting down the system now..."
# Try poweroff directly (sometimes works without sudo on single-user VMs)
poweroff

# Fallback to sudo shutdown if poweroff fails
sudo shutdown -h now
