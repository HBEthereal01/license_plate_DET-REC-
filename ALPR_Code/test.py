a= 1005
print("Value of a = ",a)
with open("value.txt", "w") as file:
    file.write(str(a))

with open("value.txt", "r") as file:
    content = file.read()
    value_a = int(content)
    print("integer value: ",value_a)              








