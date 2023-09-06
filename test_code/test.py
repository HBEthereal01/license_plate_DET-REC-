import os

pwd = os.getcwd()
print(pwd)

elements = os.listdir(pwd+"/result")

# print(elements)

for element in elements:
    print(element)


