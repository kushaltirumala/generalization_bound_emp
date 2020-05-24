mkdir experiment_save/$1
echo "Making directory in $1"
scp -r ktirumal@ml-login3.cms.caltech.edu:/home/ktirumal/weight_norm_noise/experiment_save experiment_save/$1
scp ktirumal@ml-login3.cms.caltech.edu:/home/ktirumal/weight_norm_noise/*.png experiment_save/$1
