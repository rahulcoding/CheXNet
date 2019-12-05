import os
listy = os.listdir('images')
total_len = len(listy)
#print(total_len)
train_len = int(total_len*0.90)
#print(train_len)
valid_len = int(total_len*0.05) 
new_list = []
for j,i in enumerate(listy):
    new_list.append(i+'\n')
    #listy.pop(j)
    if (j == train_len):
        break
#print(listy)
new_list[-1]= new_list[-1][:-1]
#print(len(new_list),"<-- length of train list")
data = "".join(new_list)
#print(data)
with open('train_images.txt','w') as f:
    f.write(data)


#validation
valid_list = []
bo = False
for j,i in enumerate(listy):
    if (j == train_len  or bo):
        bo = True
        valid_list.append(i+'\n')
        #listy.pop(j)
        if (j == (valid_len+train_len)):
            break
valid_list[-1] = valid_list[-1][:-1]
#print(len(valid_list))
valid_data = "".join(valid_list)
with open('valid_images.txt','w') as f:
    f.write(valid_data)

#test
test_list = []
po = False
for j,i in enumerate(listy):
    if (po or j == (valid_len+train_len)):
        po = True
        test_list.append(i+'\n')
#print(len(test_list))
test_list[-1] =  test_list[-1][:-1]
test_data = "".join(test_list)
with open('test_images.txt','w') as f:
    f.write(test_data)
