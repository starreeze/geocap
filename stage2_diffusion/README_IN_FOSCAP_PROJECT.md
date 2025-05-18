Run
```
CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU> python stage2_diffusion/process.py --rules <rules_path> --best_match <best_match_path> --kwd <save_path> --start_pos <index where rules will start here> --end_pos <index where rules will end here>
```
to manually run the generation process.

For example:
```
CUDA_VISIBLE_DEVICES=1 python stage2_diffusion/process.py --rules /home/nfs04/hanyt/geocap/dataset/rules-4-13.json --best_match /home/nfs04/hanyt/geocap/dataset/4-13/best_match.txt --kwd /home/nfs04/hanyt/geocap/dataset/4-13/pics --start_pos 0 --end_pos 100
```
You can use `jobq` to automize this process.