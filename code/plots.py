# M = N = K
import matp
class Experiment:
    def __init__(self, n_runs, elm_type, acc_type, matrix_size, register_tiled, tensor_naive, tensor_optim, cublas):
        self.n_runs = 25
        self.matrix_sizez = matrix_sizez
        self.register_tiled = register_tiled
        self.tensor_naive = tensor_naive
        self.tensor_optim = tensor_optim
        self.cublas = cublas

n_runs = 25
matrix_sizez =  [128,    256,    512,     1024,     2048,     4096,    8128]
register_tiled =[99.77,  552.61, 1945.18, 5839.36,  6701.46,  9989.81, 12714.6]
tensor_naive =  [135.65, 737.14, 2583.10, 6505.95,  9421.06,  17568.2, 17369.7]
tensor_optim =  [127.41, 700.22, 2401.89, 11623.10, 17121.70, 37243.9, 37713.9]
cublas =        [246.72, 1864.14,14708.8, 46939.50, 121413.0, 236251.0,259219]

