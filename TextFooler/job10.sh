#!/bin/sh


#cd ..

data='sms'

#python run_attack.py --data ${data} --output_dir out --target_model mlp_1  \
#--target_model_path ../binary/checkpoints/${data}_mlp_1.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model mlp_8  \
#--target_model_path ../binary/checkpoints/${data}_mlp_8.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model mlp_100  \
#--target_model_path ../binary/checkpoints/${data}_mlp_100.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model rf_100  \
#--target_model_path ../binary/checkpoints/${data}_rf_100.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model rf_8  \
#--target_model_path ../binary/checkpoints/${data}_rf_8.pkl
#
##
#python run_attack.py --data ${data} --output_dir out --target_model scdcemlp_8  \
#--target_model_path ../binary/checkpoints/${data}_scdcemlp_8_br02_h20_nr075_ni1000_i1_0.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model scdcemlp_100  \
#--target_model_path ../binary/checkpoints/${data}_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl
##
#python run_attack.py --data ${data} --output_dir out --target_model scd01mlp_8  \
#--target_model_path ../binary/checkpoints/${data}_scd01mlp_8_br02_h20_nr075_ni1000_i1_0.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model scd01mlp_100  \
#--target_model_path ../binary/checkpoints/${data}_scd01mlp_100_br02_h20_nr075_ni1000_i1_0.pkl

#python run_attack.py --data ${data} --output_dir out --target_model scdcemlpbnn_8  \
#--target_model_path ../binary/checkpoints/${data}_scdcemlpbnn_8_br02_h20_nr075_ni1000_i1_0.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model scdcemlpbnn_100  \
#--target_model_path ../binary/checkpoints/${data}_scdcemlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl
##
#python run_attack.py --data ${data} --output_dir out --target_model scd01mlpbnn_8  \
#--target_model_path ../binary/checkpoints/${data}_scd01mlpbnn_8_br02_h20_nr075_ni1000_i1_0.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model scd01mlpbnn_100  \
#--target_model_path ../binary/checkpoints/${data}_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl
#
#python run_attack.py --data ${data} --output_dir out --target_model bnn_8  \
#--target_model_path ../binary/checkpoints/${data}_mlpbnn_approx --vote 8
#
#python run_attack.py --data ${data} --output_dir out --target_model bnn_100  \
#--target_model_path ../binary/checkpoints/${data}_mlpbnn_approx --vote 100

#python run_attack.py --data ${data} --output_dir out --target_model wordCNN  \
#--target_model_path checkpoints/ag --nclasses 4

#python run_attack.py --data ${data} --output_dir out --target_model scd01mlpbnn_100_8  \
#--target_model_path ../binary/checkpoints/${data}_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --vote 8

for i in {0..5}
do
#python train_classifier.py --cnn --dataset mr --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/mr_bp_d0_b_${i} --lr 0.001
#python run_attack.py --data mr --output_dir test_out --target_model mr_bp_relu_${i} --target_model_path checkpoints/mr_wordcnn01_bp_relu_${i}.pkl
#python run_attack.py --data mr --output_dir test_out --target_model mr_bp_01_fs_${i} --target_model_path checkpoints/mr_wordcnn01_bp_01_fs_${i}.pkl --data_size 1000
#python run_attack.py --data sms --output_dir test_out --target_model sms_bp_long_01_fs_${i} --target_model_path checkpoints/sms_wordcnn01_bp_long_01_fs_${i}.pkl --data_size 1000
#python train_classifier.py --cnn --dataset ag --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/ag_bp_d0_b_${i} --lr 0.001
#python run_attack.py --data ag --output_dir test_out --target_model ag_bp_01_fs_${i} --target_model_path checkpoints/ag_wordcnn01_bp_01_${i}.pkl --data_size 1000 --nclasses 4
#python train_classifier.py --cnn --dataset imdb --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/imdb_bp_d0_b_${i} --lr 0.001
#python train_classifier.py --cnn --dataset yelp --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/yelp_bp_d0_b_${i} --lr 0.001
#python run_attack.py --data imdb --output_dir test_out --target_model imdb_bp_relu_${i} --target_model_path ../../scd/experiments/checkpoints/imdb_wordcnn01_bp_relu_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data yelp --output_dir test_out --target_model yelp_bp_relu_${i} --target_model_path ../../scd/experiments/checkpoints/yelp_wordcnn01_bp_relu_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data imdb --output_dir test_out --target_model imdb_bp_01_fs_${i} --target_model_path ../../scd/experiments/checkpoints/imdb_wordcnn01_bp_01_fs_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data yelp --output_dir test_out --target_model yelp_bp_01_fs_${i} --target_model_path ../../scd/experiments/checkpoints/yelp_wordcnn01_bp_01_fs_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data yelp --output_dir test_out --target_model yelp_bp_01_${i} --target_model_path ../../scd/experiments/checkpoints/yelp_wordcnn01_bp_01_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data imdb --output_dir test_out --target_model imdb_bp_01_${i} --target_model_path ../../scd/experiments/checkpoints/imdb_wordcnn01_bp_01_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data imdb --output_dir test_out --target_model imdb_bp_relu_fs_${i} --target_model_path ../../scd/experiments/checkpoints/imdb_wordcnn01_bp_relu_fs_${i}.pkl --data_size 1000 --nclasses 2
#python train_classifier.py --cnn --dataset imdb --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/imdb_bp_d0_b_${i} --lr 0.001
python run_attack.py --data ag --output_dir test_out --target_model ag_bp_01_fs_${i} --target_model_path ../../scd/experiments/checkpoints/ag_wordcnn01_bp_01_fs_${i}.pkl --data_size 1000 --nclasses 4
#python train_classifier.py --cnn --dataset mr --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/mr_bp_d0_b_${i} --lr 0.001
#python train_classifier.py --cnn --dataset ag --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/ag_bp_d0_b_${i} --lr 0.001
#python train_classifier.py --cnn --dataset yelp --dropout 0 --batch_size 16 --max_epoch 70 --embedding glove.6B.200d.txt --seed ${i} --save_path checkpoints/yelp_bp_d0_b_${i} --lr 0.001
#python run_attack.py --data ag --output_dir test_out --target_model ag_bp_relu_fs_${i} --target_model_path ../../scd/experiments/checkpoints/ag_wordcnn01_bp_relu_fs_${i}.pkl --data_size 1000 --nclasses 4
#python run_attack.py --data yelp --output_dir test_out --target_model yelp_bp_relu_fs_${i} --target_model_path ../../scd/experiments/checkpoints/yelp_wordcnn01_bp_relu_fs_${i}.pkl --data_size 1000 --nclasses 2
#python run_attack.py --data mr --output_dir test_out --target_model mr_bp_relu_fs_${i} --target_model_path ../../scd/experiments/checkpoints/mr_wordcnn01_bp_relu_fs_${i}.pkl --data_size 1000 --nclasses 2


done