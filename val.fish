# !/usr/bin/fish
set folder_path "/root/SparseBEV/outputs/SparseBEV/deep-supervisored-v1"
conda activate deepsupervisor
while true
    for file in (ls $folder_path/epoch_*.pth)
        set epoch_number (string match -r '\d+' (basename $file))
        set remainder (math "$epoch_number % 6")
        if test $remainder -eq 0
            set logfile "log/val_"(basename $file)".log"
            if test -e $logfile
                continue
            end
            python val.py --config configs/r50_nuimg_704x256.py --weights $file &> $logfile
        end
    end
    sleep 60  # 每60秒检查一次
end