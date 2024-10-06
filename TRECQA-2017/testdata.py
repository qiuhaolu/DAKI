from utils import *
processor = processors['regression']()
#examples_trec = processor.get_train_examples('data/') 
#examples = processor.get_test_examples('data/')
examples_both = processor.get_dev_examples('data/')
examples_test = processor.get_test_examples('data/')
#examples_test = processor.get_test_examples('data/')
print(len(examples_test)) 
print(len(examples_both))
#print(examples[0].guid)
#print(examples[0].text_a)
#print(examples[0].text_b)
#print(examples[0].label)
cnt=0
for i in range(len(examples_test)):
	test_text_a = examples_test[i].text_a
	test_text_b = examples_test[i].text_b
	for j in range(len(examples_both)):
		trainboth_text_a = examples_both[j].text_a
		trainboth_text_b = examples_both[j].text_b

		if(test_text_a == trainboth_text_a and test_text_b == trainboth_text_b):
			cnt+=1
			break

print(cnt)