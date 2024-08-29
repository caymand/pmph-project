import colorama
import numpy as np
from colorama import Back

X = 8
Y = 4

colors = {0: Back.BLUE, 1: Back.GREEN, 2: Back.YELLOW, 3: Back.RED}

def print_cell(x, y):
    print(colors[y] + f"({x*X:02}...{(x+1) * X - 1:02}, {y})", end=Back.RESET + "\t")

def print_newline():
    print(Back.RESET + "\n")

# Original
for y in range(Y):
    for x in range(X):
        print_cell(x, y)

    print(Back.RESET + "\n")

print("\n\n\n")

# Option 1
# def global_shared_map(x, y):
#     new_y = (x & 1) | ((x >> 1) & 2)
#     new_x = ((x << 1) & 4) | y ^ new_y
#     return new_x, new_y

# Option 2
def global_shared_map(x, y):
    new_y = x // 2
    row_index = (x * 4 + y) % 8
    new_x = row_index

    if new_y % 2 == 1:
        if row_index % 2 == 0:
            new_x += 1
        else:
            new_x -= 1

    # new_x = new_x ^ (y & 2)
    # TODO: use XOR instead
    if new_y // 2 != 0:
        if y // 2 != 0:
            new_x -= 2
        else:
            new_x += 2

    # new_x = row_index + (new_y % 2) * (- (row_index % 2) + (1 - (row_index % 2))) + (new_y // 2) * 2 * (-(row_index // 2) + (1 - (row_index // 2)))
    return new_x, new_y

shared_global_map = {}

for y in range(Y):
    for x in range(X):
        new_x, new_y = global_shared_map(x, y)

        shared_global_map[(new_x, new_y)] = (x, y)


for y in range(Y):
    for x in range(X):
        src_x, src_y = shared_global_map[(x, y)]
        print_cell(src_x, src_y)

    print_newline()

print("\n\n\n")

# MMA matrices
# for matrix in range(4):
#     for row in range(8):
#         row_x, row_y = shared_global_map[(row, matrix)]
#         print_cell(row_x, row_y)
#         print_newline()
#
#     print("\n\n\n")

# for row in range(4):
#     for cell in range(2):
#         row_x, row_y = (cell, row)
#         print_cell(row_x, row_y)
#     print_newline()


# Load each row as separate matrix
for matrix in range(4):
    for row in range(8):
        new_x, new_y = shared_global_map[row, matrix]
        print_cell(new_x, new_y)
        print_newline()

    print("\n\n\n")


# test = [[f"({row // 4 * 8 + col}, {row % 4})" for col in range(8)] for row in range(8)]
# print(np.array(test).T)


# for matrix in range(4):
#     for row in range(8):
#         print_cell(row, matrix)
#         print_newline()
#
#     print("\n\n\n")

