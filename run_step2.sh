#Unsupervised
python step2/step2.py -l 32 -p models/focalnetae --workers 6 --backbone FocalNet-srf -o FocalNetAE --mae --gamma 1.4 --seed 542

#Unsupervised Independent
python step2/step2_holdout.py -l 32 -p models/focalnetae --workers 6 --backbone FocalNet-srf -o FocalNetAE --mae --seed 542

#Supervised
python step2/step2_fewshot.py -l 32 -p models/focalnetae --workers 6 --backbone FocalNet-srf -o FocalNetAE --seed 542

#Supervised Independent
python step2/step2_fewshot_holdout.py -l 32 -p models/focalnetae --workers 6 --backbone FocalNet-srf -o FocalNetAE --seed 542
