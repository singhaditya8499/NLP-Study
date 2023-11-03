# NLP-Study

## Named Entity Recognition

### POS tagging

POS Tagging (Parts of Speech Tagging) is a process to mark up the words in text format for a particular part of a speech based on its definition and context. It is responsible for text reading in a language and assigning some specific token (Parts of Speech) to each word. It is also called grammatical tagging.

Source: https://www.guru99.com/pos-tagging-chunking-nltk.html

### Chunking

Chunking in NLP is a process to take small pieces of information and group them into large units. The primary use of Chunking is making groups of “noun phrases.” It is used to add structure to the sentence by following POS tagging combined with regular expressions. The resulted group of words are called “chunks.” It is also called shallow parsing.

It can be done using `nltk` library


### How to do Named Entity Recognition?

There are two ways to do it:

1. Using the nltk library

```
from nltk.chunk import ne_chunk

def extract_ne(trees, labels):
    
    ne_list = []
    for tree in ne_res:
        if hasattr(tree, 'label'):
            if tree.label() in labels:
                ne_list.append(tree)
    
    return ne_list
    
# ex is the text
            
ne_res = ne_chunk(pos_tag(word_tokenize(ex)))
labels = ['ORGANIZATION']

```


2. Using spaCy library

```
import spacy

nlp = spacy.load("en_core_web_sm")
wiki_ex = df_wikibooks.iloc[11]['body_text']
doc = nlp(wiki_ex)
doc

print('All entity types that spacy recognised from the document above')
set([ent.label_ for ent in doc.ents])

print('Persons from the document above')
print(set([ent for ent in doc.ents if ent.label_ == 'PERSON']))
print('Organizations from the document above')
print(set([ent for ent in doc.ents if ent.label_ == 'ORG']))

```

Source: https://www.kaggle.com/code/eneszvo/ner-named-entity-recognition-tutorial

### Finetuning a model

The process of training a neural network is a difficult and time-consuming process and for most of the users not even feasible. Because of that, instead of training the model from scratch, we can use models from Hugging Face which has been trained using a large amount of text.

These types of models through training developed a statistical understanding of the language they have been trained on, but they might not be useful for our specific task. In order to utilize the knowledge of the model, we can apply fine-tuning. It means that we can take pretrained model and train it a little bit more with our annotated data.

This process is called transfer learning when the knowledge is transfered from one model to another one and that strategy is often used in deep learning.

## Dependency Parsing

### What is Dependency Parsing?

Dependency Parsing is the process to analyze the grammatical structure in a sentence and find out related words as well as the type of the relationship between them.

Each relationship:

1. Has one head and a dependent that modifies the head.
2. Is labeled according to the nature of the dependency between the head and the dependent. These labels can be found at https://universaldependencies.org/u/dep/


There are different ways to do dependency parsing. The main 3 ways are:

1. Using spaCy library
2. Using standard nltk with standard Stanford CoreNLP
3. Using stanza

Source: https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7

## Sentiment Analysis

- Sentiment analysis, or opinion mining, uses natural language processing to extract subjective information from text.
- It assesses the emotional tone or attitude toward a specific topic or product in a document.
- Sentiment analysis can be performed at the document level, sentence level, or entity and aspect level.
- It is widely used for brand management, product analysis, and understanding customer sentiments.
- Various models are used for sentiment analysis, including LSTM networks, CNNs, and Transformer-based models like BERT and GPT, which are known for capturing complex contextual relationships in text.


## Text summarisation

- Text summarization aims to create concise summaries of long texts while retaining essential information and meaning.
- Two primary types of summarization are extractive, which selects key phrases and sentences from the original text, and abstractive, which generates a new summary using potentially different words and phrases.
- Sequence-to-sequence models, like those based on LSTM or GRU networks, are commonly used for text summarization. These models read the input text and produce a summary as another sequence.
- For abstractive summarization, Transformer-based models like T5 and BART have demonstrated strong performance due to their ability to understand and generate complex text.

## Question Answering

- Question answering is an NLP task that aims to provide accurate answers to human-posed questions.
- Questions can be simple factoid queries or more complex ones that require context understanding and reasoning.
- The goal is to deliver accurate, concise, and relevant responses to user queries.
- Developing question answering systems requires a deep understanding of both natural language understanding and generation.
- Transformer architectures, notably BERT and its variants, have significantly advanced question answering tasks. These models are pre-trained on extensive text data and fine-tuned for specific question answering tasks, enabling them to understand context and generate precise answers.

## Text classification

- Text classification is a fundamental NLP task that involves categorizing text into groups based on its content, with applications in spam filtering, sentiment analysis, and topic labeling.

- It makes text data more manageable and interpretable by effectively categorizing it.

- Different architectures can be used for text classification, depending on the complexity of the task. Traditional approaches include CNNs, RNNs (including LSTMs and GRUs) for capturing sequential information in text.

- For more complex tasks, Transformer-based models like BERT and XLNet are employed, leveraging self-attention mechanisms to understand the context of each word in a text, leading to superior performance.