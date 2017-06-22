# import skimage.io
# with open('poop.txt') as f:
# 	for l in f.readlines():
# 		try:
# 			skimage.io.imread(l[:-1])
# 		except IOError:
# 			print("poop") 


# import os

# for i in os.listdir('.'):
#         if i == 'check.py':
#                 continue
#         ls = os.listdir(i)
#         for j in ls:
#                 if j[-4:] != '.jpg':
#                         print(j)

import sys

print sys.argv[1]

print sys.argv[1:]