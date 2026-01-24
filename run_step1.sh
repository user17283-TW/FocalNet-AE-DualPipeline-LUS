#Finetune AE
python step1/step1.py --latent_dim 32 --seed 2000 -o focalnetae_new --loss mix --epochs 1000 --mae --backbone FocalNet_srf --with_training --loss_func hybrid -tlw 3

