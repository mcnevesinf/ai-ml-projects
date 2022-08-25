
#Adapted from: https://deepnote.com/@isaac-aderogba/Spacy-Food-Entities-2cc2d19c-c3ac-4321-8853-0bcf2ef565b3

import en_core_web_lg
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import re
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import warnings


def plot_food_len(foods):
	one_worded_foods = foods[foods.str.split().apply(len) == 1]
	two_worded_foods = foods[foods.str.split().apply(len) == 2]
	three_worded_foods = foods[foods.str.split().apply(len) == 3]
	
	#Create a bar plot
	fig, ax = plt.subplots(figsize=(10,6))
	ax.bar([1,2,3], [one_worded_foods.size, two_worded_foods.size, three_worded_foods.size])
	
	plt.show()


#Make the data set more uniform wrt food name sizes
def make_data_uniform(foods, one_word=45, two_words=30, three_words=25):
	
	print(foods.size)
	
	one_worded_foods = foods[foods.str.split().apply(len) == 1]
	two_worded_foods = foods[foods.str.split().apply(len) == 2]
	three_worded_foods = foods[foods.str.split().apply(len) == 3]
	
	#Total number of foods
	total_num_foods = round(one_worded_foods.size / 45 * 100)
	
	#Shuffle 2-worded and 3-worded foods as they will be sliced
	two_worded_foods = two_worded_foods.sample(frac=1)
	three_worded_foods = three_worded_foods.sample(frac=1)
	
	#Append the foods together
	foods = pd.concat([one_worded_foods, two_worded_foods[:round(total_num_foods* (two_words/100))]])
	foods = pd.concat([foods, three_worded_foods[:round(total_num_foods * (three_words/100))]])
	
	return foods


def generate_food_sentences(foods, num_sentences=500):
	
	food_templates = [
    "I ate my {}",
    "I'm eating a {}",
    "I just ate a {}",
    "I only ate the {}",
    "I'm done eating a {}",
    "I've already eaten a {}",
    "I just finished my {}",
    "When I was having lunch I ate a {}",
    "I had a {} and a {} today",
    "I ate a {} and a {} for lunch",
    "I made a {} and {} for lunch",
    "I ate {} and {}",
    "today I ate a {} and a {} for lunch",
    "I had {} with my husband last night",
    "I brought you some {} on my birthday",
    "I made {} for yesterday's dinner",
    "last night, a {} was sent to me with {}",
    "I had {} yesterday and I'd like to eat it anyway",
    "I ate a couple of {} last night",
    "I had some {} at dinner last night",
    "Last night, I ordered some {}",
    "I made a {} last night",
    "I had a bowl of {} with {} and I wanted to go to the mall today",
    "I brought a basket of {} for breakfast this morning",
    "I had a bowl of {}",
    "I ate a {} with {} in the morning",
    "I made a bowl of {} for my breakfast",
    "There's {} for breakfast in the bowl this morning",
    "This morning, I made a bowl of {}",
    "I decided to have some {} as a little bonus",
    "I decided to enjoy some {}",
    "I've decided to have some {} for dessert",
    "I had a {}, a {} and {} at home",
    "I took a {}, {} and {} on the weekend",
    "I ate a {} with {} and {} just now",
    "Last night, I ate an {} with {} and {}",
    "I tasted some {}, {} and {} at the office",
    "There's a basket of {}, {} and {} that I consumed",
    "I devoured a {}, {} and {}",
    "I've already had a bag of {}, {} and {} from the fridge"
	]
	
	food_data = {
		"one_food" : [],
		"two_foods" : [],
		"three_foods" : []
	}
	
	
	#Pattern to replace from the template sentences
	pattern_to_replace = "{}"
	
	#Shuffle the data before starting
	foods = foods.sample(frac=1)
	
	num_sentences_count = 1
	
	while num_sentences_count < num_sentences:
	
		#Will contain the (food) entities on each sentence
		entities = []
	
		#Pick a random food template
		sentence = food_templates[random.randint(0, len(food_templates)-1)]
		
		#Find out how many braces "{}" need to be replaced in the template
		matches = re.findall(pattern_to_replace, sentence)
		
		#For each brace, replace with a food entity from the shuffled food data
		for match in matches:
			food = foods.iloc[random.randint(0, foods.size-1)]

			#Replace the pattern
			sentence = sentence.replace(match, food, 1)
			
			#Find the match of the food entity we just inserted
			match_span = re.search(food, sentence).span()
			
			#Append the index positions of the food entity we just inserted
			entities.append( (match_span[0], match_span[1], "FOOD") )
			
		#Append the sentence and the position of the entities to the correct array
		if len(matches) == 1:
			food_data["one_food"].append( (sentence, {"entities" : entities}) )
		elif len(matches) == 2:
			food_data["two_foods"].append( (sentence, {"entities" : entities}) )
		else:
			food_data["three_foods"].append( (sentence, {"entities" : entities}) )
			
		num_sentences_count += 1
	
	return food_data	
	

def generate_revision_sentences(rawData, num_articles=500):
	
	revision_texts = []
	revision_data = []
	
	#Create a spacy nlp object to identify existing sentences
	nlp = en_core_web_lg.load()
	
	#Convert the articles to spacy objects to identify the sentences. Disable unneeded components
	for doc in nlp.pipe(npr_df["Article"][:num_articles], batch_size=30, disable=["ner"]):
		for sentence in doc.sents:
			#Keep sentences of a similar length to the generated food sentences		
			if 40 < len(sentence.text) < 80:
				#Some of the sentences have excessive whitespace in between words, 
				#so we're trimming that
				revision_texts.append(" ".join(re.split("\s+", sentence.text, flags=re.UNICODE)))

	#Predict the entities, then append the entry to the revision data set
	for doc in nlp.pipe(revision_texts, batch_size=30, disable=["parser"]):
	
		#Don't append sentences that have no entities
		if len(doc.ents) > 0:
			revision_data.append((doc.text, {"entities" : [(e.start_char, e.end_char, e.label) for e in doc.ents]}))
			
	return revision_data
	

def train(train_data, model, epochs=3):
	
	#Add the new label to the NER model
	ner = nlp.get_pipe("ner")
	ner.add_label("FOOD")
	
	#Get the names of the components we want to disable during training
	pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
	
	#start the training loop, only training NER
	optimizer = nlp.resume_training()
	
	with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():

		warnings.filterwarnings("once", category=UserWarning, module='spacy')
		sizes = compounding(1.0, 4.0, 1.001)
		
		#Batch up the examples using spacy's minibatch
		for epoch in range(epochs):
			examples = train_data
			random.shuffle(examples)
			batches = minibatch(examples, size=sizes)
			losses = {}
			
			for batch in batches:
				texts, annotations = zip(*batch)
				
				#Create example
				examples = []
				
				for i in range(len(texts)):
					doc = nlp.make_doc(texts[i])
					examples.append(Example.from_dict(doc, annotations[i]))
				
				nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
				
			print("Losses ({}/{})".format(epoch+1, epochs), losses)



if __name__ == '__main__':

	#Read the food data set
	food_df = pd.read_csv("data/food.csv")

	#Remove foods with special characters, lowercase and extract results from 'description' column
	foods = food_df[food_df["description"].str.contains("[^a-zA-Z ]") == False]["description"].apply(lambda food : food.lower())

	#Filter out foods woth more than 3 words, drop any duplicates
	foods = foods[foods.str.split().apply(len) <= 3].drop_duplicates()

	foods = make_data_uniform(foods)

	train_food_data = generate_food_sentences(foods, 1000)

	for key in train_food_data:
		print("{} {} sentences".format(len(train_food_data[key]), key))

	test_food_data = generate_food_sentences(foods)

	for key in train_food_data:
		print("{} {} sentences".format(len(test_food_data[key]), key))

	#Preparing the revision data (used to avoid catastrophic forgetting)
	npr_df = pd.read_csv("data/articles.csv")
	
	train_revision_data = generate_revision_sentences(npr_df, num_articles=1000)

	#Combine both training data sets (food + revision)
	train_data = train_food_data["one_food"] + train_food_data["two_foods"] + train_food_data["three_foods"] + train_revision_data

	print("REVISION ", len(train_revision_data))
	print("COMBINED ", len(train_data))

	nlp = en_core_web_lg.load()

	train(train_data, nlp)

	#Save trained model
	nlp.meta["name"] = "food_entity_extractor_v2"
	
	if not os.path.exists("./trained-model/"):
		os.mkdir("./trained-model/")
	
	nlp.to_disk("./trained-model/")


























