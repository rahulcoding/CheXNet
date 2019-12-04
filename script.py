import os
listy = os.listdir('/content/')
total_len = len(listy)
train_len = int(total_len*0.80)
valid_len = int(total_len*0.10) 
new_list = []
for j,i in enumerate(listy):
  new_list.append(i+'\n')
  listy.pop(j)
  if (j == train_len):
    break
print(listy)
new_list[-1]= new_list[-1][:-1]
print(new_list)
data = "".join(new_list)
print(data)
with open('train_images.txt','w') as f:
  f.write(data)


#validation
valid_list = []
for j,i in enumerate(listy):
  valid_list.append(i+'\n')
  listy.pop(j)
  if (j == valid_len):
    break
valid_list[-1] = valid_list[-1][:-1]
valid_data = "".join(valid_list)
with open('valid_images.txt','w') as f:
  f.write(valid_data)

#test
test_list = []
for i in listy:
  test_list.append(i+'\n')
test_list[-1] =  test_list[-1][:-1]
test_data = "".join(test_list)
with open('test_images.txt','w') as f:
  f.write(test_data)
