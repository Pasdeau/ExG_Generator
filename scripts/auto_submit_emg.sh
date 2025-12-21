#!/bin/bash
# Auto-submit EMG training after wget download completes

CHECK_INTERVAL=60  # 1 minute

echo "Waiting for wget download to complete..."

while true; do
    # Check if wget is still running
    WGET_RUNNING=$(pgrep -u wenwang wget 2>/dev/null | wc -l)
    CURRENT_SIZE=$(du -sh ~/exg_train/physionet.org/ 2>/dev/null | cut -f1)
    CURRENT_FILES=$(find ~/exg_train/physionet.org -type f 2>/dev/null | wc -l)
    
    echo "$(date): wget processes: $WGET_RUNNING, files: $CURRENT_FILES ($CURRENT_SIZE)"
    
    if [ "$WGET_RUNNING" -eq 0 ]; then
        echo ""
        echo "wget download completed! Submitting EMG training job..."
        cd ~/exg_train
        sbatch slurm_emg_train.sh
        echo "EMG training job submitted!"
        exit 0
    fi
    
    sleep $CHECK_INTERVAL
done
