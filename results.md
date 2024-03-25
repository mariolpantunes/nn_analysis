## Aggregations

| | **All Datasets** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 21.00 | 0.25 | 0.20 | 
| learning_rate | 7.66 | 8.36 | 7.99 | 
| loss | 0.59 | 0.24 | 0.19 | 
| batch_size | 2.25 | 43.25 | 37.16 | 
 
| | **Dense** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 15.11 | 0.25 | 0.16 | 
| learning_rate | 8.27 | 10.58 | 9.61 | 
| loss | 0.54 | 0.31 | 0.23 | 
| batch_size | 2.12 | 44.44 | 41.08 | 
| n_dense_nodes | 2.27 | 10.24 | 0.62 | 
| dense_activation | 4.14 | 0.40 | 1.35 | 
| dense_dropout_rate | 3.48 | 0.25 | 0.36 | 
| n_dense_layers | 8.83 | 10.48 | 27.56 | 

| | **Dense Regression** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 0.87 | 0.37 | 0.19 | 
| learning_rate | 10.08 | 6.37 | 3.06 | 
| loss | 1.30 | 0.54 | 0.53 | 
| batch_size | 2.11 | 32.89 | 30.31 | 
| n_dense_nodes | 2.40 | 14.17 | 0.70 | 
| dense_activation | 4.03 | 0.65 | 1.84 | 
| dense_dropout_rate | 6.26 | 0.31 | 0.76 | 
| n_dense_layers | 1.75 | 15.99 | 39.17 | 

 

| | **Dense Classification** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 22.23 | 0.19 | 0.15 | 
| learning_rate | 7.37 | 12.69 | 12.88 | 
| loss | 0.16 | 0.19 | 0.08 | 
| batch_size | 2.13 | 50.22 | 46.47 | 
| n_dense_nodes | 2.21 | 8.28 | 0.58 | 
| dense_activation | 4.19 | 0.27 | 1.11 | 
| dense_dropout_rate | 2.09 | 0.23 | 0.16 | 
| n_dense_layers | 12.36 | 7.72 | 21.76 | 

 

| | **CNN Classification** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 38.68 | 0.26 | 0.30 | 
| learning_rate | 5.83 | 1.69 | 3.13 | 
| loss | 0.73 | 0.04 | 0.06 | 
| batch_size | 2.62 | 39.69 | 25.38 | 
| n_kernels | 1.74 | 13.04 | 12.71 | 
| kernel_size | 0.50 | 0.56 | 0.25 | 
| pool_size | 2.43 | 0.53 | 0.45 | 
| conv_activation | 0.26 | 0.04 | 0.14 | 
| conv_dropout_rate | 1.36 | 0.17 | 0.93 | 
| n_conv_layers | 1.52 | 10.20 | 20.02 | 
| n_dense_nodes | 1.66 | 3.29 | 2.58 | 
| dense_activation | 1.12 | 0.22 | 0.77 | 
| dense_dropout_rate | 2.00 | 0.66 | 0.35 | 
| n_dense_layers | 2.37 | 1.12 | 4.36 | 

## Individual models

| | **abalone DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 0.26 | 0.46 | 0.14 | 
| learning_rate | 17.13 | 2.72 | 2.36 | 
| loss | 0.96 | 0.15 | 0.23 | 
| n_dense_nodes | 0.19 | 23.84 | 0.54 | 
| dense_activation | 3.81 | 0.68 | 4.68 | 
| dense_dropout_rate | 2.43 | 0.54 | 1.38 | 
| n_dense_layers | 2.91 | 30.64 | 61.22 | 
| batch_size | 1.25 | 2.31 | 2.20 | 

 

| | **bike_sharing DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 1.61 | 0.22 | 0.36 | 
| learning_rate | 12.89 | 5.65 | 2.12 | 
| loss | 1.23 | 1.36 | 0.30 | 
| n_dense_nodes | 1.59 | 18.19 | 1.10 | 
| dense_activation | 5.04 | 0.89 | 0.82 | 
| dense_dropout_rate | 8.32 | 0.18 | 0.42 | 
| n_dense_layers | 2.02 | 17.17 | 56.03 | 
| batch_size | 0.95 | 18.60 | 13.13 | 

 

| | **compas DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 8.00 | 0.70 | 0.22 | 
| learning_rate | 6.73 | 5.96 | 0.76 | 
| loss | 0.15 | 0.83 | 0.20 | 
| n_dense_nodes | 5.27 | 19.37 | 0.55 | 
| dense_activation | 6.98 | 0.72 | 4.12 | 
| dense_dropout_rate | 2.98 | 0.75 | 0.21 | 
| n_dense_layers | 16.72 | 27.20 | 67.14 | 
| batch_size | 2.07 | 3.78 | 5.37 | 

 

| | **covertype DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 11.06 | 0.04 | 0.05 | 
| learning_rate | 7.05 | 15.17 | 3.60 | 
| loss | 0.28 | 0.16 | 0.02 | 
| n_dense_nodes | 1.31 | 2.88 | 0.02 | 
| dense_activation | 13.03 | 0.46 | 0.52 | 
| dense_dropout_rate | 1.27 | 0.06 | 0.18 | 
| n_dense_layers | 6.63 | 2.00 | 1.31 | 
| batch_size | 3.11 | 67.32 | 83.60 | 

 

| | **delays_zurich DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 0.73 | 0.43 | 0.07 | 
| learning_rate | 0.22 | 10.73 | 4.71 | 
| loss | 1.70 | 0.11 | 1.06 | 
| n_dense_nodes | 5.42 | 0.47 | 0.47 | 
| dense_activation | 3.26 | 0.37 | 0.02 | 
| dense_dropout_rate | 8.03 | 0.21 | 0.47 | 
| n_dense_layers | 0.33 | 0.17 | 0.24 | 
| batch_size | 4.12 | 77.77 | 75.61 | 

 

| | **higgs DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 17.90 | 0.02 | 0.07 | 
| learning_rate | 9.83 | 23.75 | 31.01 | 
| loss | 0.20 | 0.04 | 0.11 | 
| n_dense_nodes | 0.67 | 3.75 | 0.97 | 
| dense_activation | 1.09 | 0.03 | 0.19 | 
| dense_dropout_rate | 5.12 | 0.02 | 0.04 | 
| n_dense_layers | 18.32 | 1.67 | 0.20 | 
| batch_size | 2.18 | 59.03 | 58.29 | 

 

| | **cifar10 CNNModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 49.11 | 0.28 | 0.46 | 
| learning_rate | 2.78 | 2.08 | 1.73 | 
| loss | 1.50 | 0.02 | 0.05 | 
| n_kernels | 0.57 | 37.15 | 36.46 | 
| kernel_size | 1.21 | 0.28 | 0.22 | 
| pool_size | 0.09 | 0.18 | 0.07 | 
| conv_activation | 0.07 | 0.04 | 0.14 | 
| conv_dropout_rate | 1.07 | 0.26 | 0.53 | 
| n_conv_layers | 1.34 | 28.30 | 25.74 | 
| n_dense_nodes | 0.78 | 0.68 | 1.23 | 
| dense_activation | 0.34 | 0.02 | 0.08 | 
| dense_dropout_rate | 1.28 | 0.88 | 0.19 | 
| n_dense_layers | 0.22 | 0.10 | 0.30 | 
| batch_size | 3.17 | 1.39 | 4.87 | 

 

| | **cifar10 DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 33.29 | 0.14 | 0.07 | 
| learning_rate | 7.83 | 10.47 | 8.74 | 
| loss | 0.20 | 0.06 | 0.01 | 
| n_dense_nodes | 3.08 | 8.81 | 0.67 | 
| dense_activation | 2.62 | 0.16 | 1.12 | 
| dense_dropout_rate | 1.39 | 0.12 | 0.24 | 
| n_dense_layers | 4.70 | 5.17 | 25.25 | 
| batch_size | 1.13 | 58.06 | 45.80 | 

 

| | **mnist CNNModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 37.83 | 0.32 | 0.15 | 
| learning_rate | 5.71 | 2.54 | 1.74 | 
| loss | 0.24 | 0.09 | 0.10 | 
| n_kernels | 1.57 | 1.30 | 1.26 | 
| kernel_size | 0.15 | 0.15 | 0.39 | 
| pool_size | 1.48 | 1.01 | 0.55 | 
| conv_activation | 0.27 | 0.04 | 0.23 | 
| conv_dropout_rate | 2.35 | 0.18 | 0.85 | 
| n_conv_layers | 2.21 | 0.81 | 16.61 | 
| n_dense_nodes | 3.27 | 3.61 | 3.58 | 
| dense_activation | 2.37 | 0.02 | 1.93 | 
| dense_dropout_rate | 2.20 | 0.18 | 0.71 | 
| n_dense_layers | 2.61 | 1.59 | 6.54 | 
| batch_size | 1.30 | 63.41 | 32.30 | 

 

| | **mnist DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 32.06 | 0.16 | 0.27 | 
| learning_rate | 5.30 | 11.44 | 13.95 | 
| loss | 0.13 | 0.03 | 0.12 | 
| n_dense_nodes | 0.88 | 7.31 | 0.86 | 
| dense_activation | 0.32 | 0.17 | 0.56 | 
| dense_dropout_rate | 1.16 | 0.29 | 0.12 | 
| n_dense_layers | 16.09 | 7.00 | 16.72 | 
| batch_size | 1.83 | 49.98 | 43.53 | 

 

| | **fashion_mnist CNNModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 29.10 | 0.17 | 0.28 | 
| learning_rate | 9.01 | 0.46 | 5.92 | 
| loss | 0.45 | 0.01 | 0.02 | 
| n_kernels | 3.09 | 0.68 | 0.41 | 
| kernel_size | 0.13 | 1.25 | 0.14 | 
| pool_size | 5.72 | 0.41 | 0.72 | 
| conv_activation | 0.44 | 0.04 | 0.06 | 
| conv_dropout_rate | 0.65 | 0.08 | 1.40 | 
| n_conv_layers | 1.01 | 1.50 | 17.72 | 
| n_dense_nodes | 0.92 | 5.58 | 2.93 | 
| dense_activation | 0.64 | 0.62 | 0.31 | 
| dense_dropout_rate | 2.53 | 0.92 | 0.16 | 
| n_dense_layers | 4.27 | 1.68 | 6.25 | 
| batch_size | 3.38 | 54.25 | 38.97 | 

 

| | **fashion_mnist DenseModel** | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| optimizer | 31.06 | 0.11 | 0.24 | 
| learning_rate | 7.46 | 9.34 | 19.24 | 
| loss | 0.02 | 0.03 | 0.03 | 
| n_dense_nodes | 2.05 | 7.58 | 0.44 | 
| dense_activation | 1.07 | 0.10 | 0.15 | 
| dense_dropout_rate | 0.59 | 0.11 | 0.14 | 
| n_dense_layers | 11.72 | 3.24 | 19.96 | 
| batch_size | 2.45 | 63.14 | 42.21 | 