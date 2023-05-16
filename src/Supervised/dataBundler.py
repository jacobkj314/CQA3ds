'''
    This script takes the json from a CONDAQA dataset partition and groups it by ORIGINAL PASSAGE, then QUESTION, then EDIT
    The output is a list of( (each original passage's) list of( question and list of (edited passages and editID and label)) )
'''

import re, json
from sys import argv
from random import shuffle

def same(items):
	'''
	A helper method to avoid getting bundles consisting of all the same answer
	'''
	if len(items) <= 1:
		return True
	return all(i==items[0] for i in items)

def sortByQuestion(gathered):
	questions = set(d["question"] for d in gathered)
	return [
			{
				"question":q, 
				"contexts":[
							{
								"passage":e["passage"], 
								"editID":e["editID"], 
								"label":e["label"]
							} 
	 						for e in gathered 
							if e["question"] == q
						   ]
			} 
	 			for q in questions
		   ]


def json2data(url, sortBy=sortByQuestion):
	with open(url, "r") as file:
		text = file.read()
	jsonArray = "[" + re.sub("\n", ",", text[:-1]) + "]" #format as json array
	jsonData = json.loads(jsonArray)
	
	passageIDs = set(d["PassageID"] for d in jsonData)
	
	def gather(pId):
		return [
					{
						"editID":d["PassageEditID"], 
						"passage":d['sentence1'], 
						"question":d['sentence2'], 
						"label":d['label']
					} 
					for d in jsonData 
					if d["PassageID"] == pId
			   ]
	
	return [sortBy(gather(i)) for i in passageIDs]

def json2bundles(url):
	data = json2data(url)
	bundles = []

	for passage in data:
		curr_bundles = []
		for bundle in passage:
			curr_bundles.append(
				{
					"input": [(bundle['question'] + '\n' + context['passage']) for context in bundle['contexts']],
					"answer": [(context['label']) for context in bundle['contexts']]
				}
			)

		'''
		This is one more attempt to bundle the data in a helpful way:

		bad example:
		{
		Q1 : Yes No idk
		Q2 : No ~~No~~ 
		D3 : yes idk
		}

		good example:
		{
		Q1 : Yes No idk
		Q2 : No Yes
		D3 : yes idk
		}  

		i.e., within a bundle, there CAN BE repeat answers, but not within a question

		'''
		if '-mqfa3' in argv:
			filtered_bundles = []
			for bun in curr_bundles:
				if True:# # # while not same(bun['answer']): # # # 
					# # #next_bun_in = []; next_bun_ans = [] # # #
					bun_in = bun['input']; bun_ans = bun['answer']
					filt_in = []; filt_ans = []
					ansset = set()
					queAns = list(zip(bun_in, bun_ans)); # # # shuffle(queAns) # # #
					for que, ans in queAns:
						if ans not in ansset:
							ansset.add(ans)
							filt_in.append(que)
							filt_ans.append(ans)
						'''else: # # #
							next_bun_in.append(que) # # #
							next_bun_ans.append(ans) # # #'''
					# # # bun = {"input":next_bun_in, "answer":next_bun_ans}# # #
					filtered_bundles.append({"input":filt_in, "answer":filt_ans})
			curr_bundles = filtered_bundles
			argv.append('-mq')


		'''
		Without this flag, a bundle will consist of a single question paired with a wikipedia sample and its edits, as well as the corresponding answers
		With this option, it means that, for each original wikipedia text sample, ALL questions about that text sample and its edits can occur in the same bundle
		(This can result in large bundles and possible out-of-memory errors when the -fa flag is not also used)
		'''
		if '-mq' in argv: #multi-question
			None # merge the bundles together
			complete_input = []; complete_answer = []
			for b in curr_bundles:
				complete_input.extend(b['input']); complete_answer.extend(b['answer'])
			curr_bundles = [{"input":complete_input, "answer":complete_answer}]


		'''
		Without this flag, a bundle can contain instances that have the same answer.
		With this flag, a bundle will be broken into multiple bundles, where each instance in a given bundle has a unique answer. 
			This is accomplished by shuffling the instances, and then iteratively forming bundles with unique answers.
			I.E.: if the set of instances' answers are 											{yes, no, don't know, yes, no, no, no, yes, don't know}, then:
				the first bundle will have as answers {yes, no, don't know}, leaving behind 	{___, __, __________, yes, no, no, no, yes, don't know}
				the second bundle will have as answers {yes, no, don't know}, leaving behind 	{___, __, __________, ___, __, no, no, yes, __________}
				the third bundle will have as answers {no, yes}, leaving behind 				{___, __, __________, ___, __, __, no, ___, __________}
				and the remaining instance answered "no" will not be bundled
		'''
		if '-fa' in argv: #filter-answers
			filtered_bundles = []
			for bun in curr_bundles:
				while not same(bun['answer']): # # # 
					next_bun_in = []; next_bun_ans = [] # # #
					bun_in = bun['input']; bun_ans = bun['answer']
					filt_in = []; filt_ans = []
					ansset = set()
					queAns = list(zip(bun_in, bun_ans)); shuffle(queAns) # # #
					for que, ans in queAns:
						if ans not in ansset:
							ansset.add(ans)
							filt_in.append(que)
							filt_ans.append(ans)
						else: # # #
							next_bun_in.append(que) # # #
							next_bun_ans.append(ans) # # #
					bun = {"input":next_bun_in, "answer":next_bun_ans}# # #
					filtered_bundles.append({"input":filt_in, "answer":filt_ans})
			curr_bundles = filtered_bundles
		
		if '-mqfa3' in argv:# PART II of this option - to make the bundles smaller to fit in memory
			sized_bundles = []
			for bun in curr_bundles:
				sized_bundles.append({"input":bun['input'][:6], "answer":bun['answer'][:6]})
			curr_bundles = sized_bundles
			
		bundles.extend(curr_bundles)

	return bundles

def bundle(source, destination):
	bundles = json2bundles(source)

	with open(destination, "w") as out:
		out.write('')
	with open(destination, "a") as out:
		for bundle in bundles:
			out.write(json.dumps(bundle) + '\n')

bundle("data/condaqa_train.json", "data/unifiedqa_formatted_data/condaqa_train_unifiedqa.json")

