#!/bin/sh


/home/y/yx277/Anaconda3/anaconda3/bin/python ../../hinge/TextFooler/attack_demo.py --data mr --output_dir test_out --target_model mr_bp_01_fs --target_model_path checkpoints/mr_wordcnn01_bp_relu_2.pkl --data_size 1000 --nclasses 2  --word_embeddings_path .vector_cache/glove.6B.200d.txt --counter_fitting_cos_sim_path ../../hinge/TextFooler/cos_sim_counter_fitting.npy --counter_fitting_embeddings_path ../../hinge/TextFooler/counter-fitted-vectors.txt


