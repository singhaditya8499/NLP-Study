# NLP-Study

## Problems in NLP

### Named Entity Recognition

#### POS tagging

POS Tagging (Parts of Speech Tagging) is a process to mark up the words in text format for a particular part of a speech based on its definition and context. It is responsible for text reading in a language and assigning some specific token (Parts of Speech) to each word. It is also called grammatical tagging.

Source: https://www.guru99.com/pos-tagging-chunking-nltk.html

#### Chunking

Chunking in NLP is a process to take small pieces of information and group them into large units. The primary use of Chunking is making groups of “noun phrases.” It is used to add structure to the sentence by following POS tagging combined with regular expressions. The resulted group of words are called “chunks.” It is also called shallow parsing.

It can be done using `nltk` library


#### How to do Named Entity Recognition?

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

#### Finetuning a model

The process of training a neural network is a difficult and time-consuming process and for most of the users not even feasible. Because of that, instead of training the model from scratch, we can use models from Hugging Face which has been trained using a large amount of text.

These types of models through training developed a statistical understanding of the language they have been trained on, but they might not be useful for our specific task. In order to utilize the knowledge of the model, we can apply fine-tuning. It means that we can take pretrained model and train it a little bit more with our annotated data.

This process is called transfer learning when the knowledge is transfered from one model to another one and that strategy is often used in deep learning.

### Dependency Parsing

#### What is Dependency Parsing?

Dependency Parsing is the process to analyze the grammatical structure in a sentence and find out related words as well as the type of the relationship between them.

Each relationship:

1. Has one head and a dependent that modifies the head.
2. Is labeled according to the nature of the dependency between the head and the dependent. These labels can be found at https://universaldependencies.org/u/dep/


There are different ways to do dependency parsing. The main 3 ways are:

1. Using spaCy library
2. Using standard nltk with standard Stanford CoreNLP
3. Using stanza

Source: https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7

### Sentiment Analysis

- Sentiment analysis, or opinion mining, uses natural language processing to extract subjective information from text.
- It assesses the emotional tone or attitude toward a specific topic or product in a document.
- Sentiment analysis can be performed at the document level, sentence level, or entity and aspect level.
- It is widely used for brand management, product analysis, and understanding customer sentiments.
- Various models are used for sentiment analysis, including LSTM networks, CNNs, and Transformer-based models like BERT and GPT, which are known for capturing complex contextual relationships in text.


### Text summarisation

- Text summarization aims to create concise summaries of long texts while retaining essential information and meaning.
- Two primary types of summarization are extractive, which selects key phrases and sentences from the original text, and abstractive, which generates a new summary using potentially different words and phrases.
- Sequence-to-sequence models, like those based on LSTM or GRU networks, are commonly used for text summarization. These models read the input text and produce a summary as another sequence.
- For abstractive summarization, Transformer-based models like T5 and BART have demonstrated strong performance due to their ability to understand and generate complex text.

### Question Answering

- Question answering is an NLP task that aims to provide accurate answers to human-posed questions.
- Questions can be simple factoid queries or more complex ones that require context understanding and reasoning.
- The goal is to deliver accurate, concise, and relevant responses to user queries.
- Developing question answering systems requires a deep understanding of both natural language understanding and generation.
- Transformer architectures, notably BERT and its variants, have significantly advanced question answering tasks. These models are pre-trained on extensive text data and fine-tuned for specific question answering tasks, enabling them to understand context and generate precise answers.

### Text classification

- Text classification is a fundamental NLP task that involves categorizing text into groups based on its content, with applications in spam filtering, sentiment analysis, and topic labeling.

- It makes text data more manageable and interpretable by effectively categorizing it.

- Different architectures can be used for text classification, depending on the complexity of the task. Traditional approaches include CNNs, RNNs (including LSTMs and GRUs) for capturing sequential information in text.

- For more complex tasks, Transformer-based models like BERT and XLNet are employed, leveraging self-attention mechanisms to understand the context of each word in a text, leading to superior performance.


## Preprocessing

It is a very important step that helps to reduce the complexity of the raw text and helps the in future tasks. Proper care should be taken in the preprocessing as it might also lead to loss of important information. 

### Stemming

It is used to transform the word into its most basic form. Eg. jumping, jumps and jumped can be transformed to its stemmed form "jump". This reduces the total number of words that are required to be stored in the corpus. It might also lead to errors in the output, depends on the problem that we are trying to solve.

### Lemmatization

It is similar to stemming that brings the word to their base form. The only difference being that lemmatization uses the morphological analysis of the word. The word "better" has "good" as its base word which will be correctly identified by lemmatization but not by stemming. It is computationally expensive.

### Stopwords

These are the words that are pretty common in the language and dont carry any special weight in the language related tasks like a, an, the, is, in ,etc. Although, they might play a significant role in some tasks like sentiment analysis, so whether to remove stop words should depend on the task as hand.

### Tokenization

This is generally the first step in the NLP task. It breaks the textual data in token which may be words or sentences or something else (the basic unit in that problem). 

### Lowercase

In this the textual data is completely taken to the lower case. It helps to reduce the corpus, like the words "India", "INDIA" will be converted to "india".

### Punctuation removal

Removing punctuation from the data. It might provide additional context in different problem type. It must only be removed from the data if it is not required at all.

### Spell correction

Spell check and correction are essential for identifying and fixing typos and spelling errors in text data. This process helps reduce redundancy by ensuring that words like "speling" and "spelling" are recognized as the same word after correction.

### Noise removal

Noise removal is the process of eliminating unwanted characters, digits, and text fragments that can disrupt text analysis. This can include removing headers, footers, HTML, and XML content.

### Text normalization

Text normalization involves standardizing text by converting it to the same case (usually lowercase), removing punctuation, and converting numbers to their word forms.

### POS tagging

Part-of-speech tagging identifies the grammatical category (noun, verb, adjective, etc.) of each word in a sentence. It is valuable for understanding sentence structure and aids in tasks like named entity recognition and question answering.


## Tokenizer

Breaking down of the data in its lowest form is called tokenization. The smallest unit is called token. These are the input units to the BERT algorithm.

### Ways for tokenization

1. Remove spaces between the words
2. Add special characters between the words
3. Split the string using the spaces

Since computer cant understand the text, it is important to represent these words as vectors. These vectors store the syntactic and semantic information about the words.

Subword tokenization is the tokenization on the next level. Most of the words are made of the suffixes and prefixes, these constitute OOV(out of vocabulary) words. Words like anytime can be broken to "any" and "time" which reduces the compelxity of the data and helps in understand of the text in detail.

Below are the main algorithms for sub word tokenization:

### Byte pair encoding

The process is used to tokenize the words that are morphologically rich and have complex language structure. The process looks for adjacent characters that occurs frequently and forms a word pair out of it. This happens for a number of iteration or until some conditions are met. These new words are then used to tokenize the original textual data we have. 

### Wordpiece

The WordPiece algorithm is a subword tokenization technique commonly used in natural language processing (NLP) and machine learning, particularly in models like BERT. WordPiece tokenization breaks down words into smaller subword units to handle complex languages, morphology, and unseen words effectively. Here's how the WordPiece algorithm works:

1. **Initialization**: Start with a basic vocabulary that includes individual characters, special symbols (e.g., [CLS] and [SEP] for BERT), and a few whole words. The vocabulary can be quite small initially.

2. **Tokenization**: Begin by splitting text into individual characters or subword units. The goal is to represent text as a sequence of these smaller units.

3. **Training**: The WordPiece algorithm iteratively updates the vocabulary to include frequently occurring subword units. Here's how this process typically works:
   
   a. **Frequency Analysis**: Analyze a training corpus to identify the most frequent subword units.
   
   b. **Merging**: Merge the most frequent subword units into a single new token. For example, if "un" and "breakable" are frequent, they might be merged into "unbreakable."
   
   c. **Update Vocabulary**: Add the newly merged token to the vocabulary.

4. **Repeat**: Steps 3a-3c are repeated for a set number of iterations or until a predefined stopping condition is met.

5. **Final Vocabulary**: After the training process is complete, you have a vocabulary that includes not only individual characters but also subword units that represent frequently occurring character combinations and subwords.

6. **Tokenization with Final Vocabulary**: During text tokenization for NLP tasks, the WordPiece algorithm uses the final vocabulary to segment text into subword units. This allows the model to handle complex words and out-of-vocabulary words effectively.

The WordPiece algorithm is flexible and adapts to the specific language and text data it is applied to. It is particularly useful for languages with complex morphologies, compound words, and agglutinative languages, where breaking down words into subword units helps in handling unseen or rare words. It is commonly used in transformer-based models like BERT and GPT-2, contributing to their effectiveness in various NLP tasks.


### Unigram subword tokenization

Unigram subword tokenization is a technique used in natural language processing (NLP) for breaking down words into their individual character-level components. Unlike other subword tokenization methods, unigram subword tokenization treats each character as a separate token. Here's how it works:

1. **Character-Level Tokenization**: In unigram subword tokenization, each character in a word is considered as a separate token. For example, the word "apple" would be tokenized into five tokens: 'a', 'p', 'p', 'l', 'e'.

2. **No Merging or Combinations**: Unlike other subword tokenization methods like WordPiece or Byte Pair Encoding (BPE), unigram subword tokenization does not involve merging characters or creating subword units. Each character stands on its own as an individual token.

3. **Use Cases**:
   - Unigram subword tokenization can be helpful in certain situations, especially for languages with simple morphologies and phonetic languages where characters often represent individual syllables or phonemes.
   - It is less common in modern NLP models compared to techniques like BPE or WordPiece, but it can be used in specific scenarios where character-level analysis is required.

4. **Advantages and Disadvantages**:
   - Advantages:
     - It preserves the integrity of individual characters, which can be useful in scenarios where character-level information is crucial, such as handwriting recognition or certain linguistic analyses.
     - It can be applied to languages with straightforward character-to-sound correspondence.
   - Disadvantages:
     - For many NLP tasks and languages, it results in an extremely large vocabulary, making it less practical in terms of memory and computational resources.
     - It doesn't handle complex word structures or morphological variations as effectively as other subword tokenization methods.

In summary, unigram subword tokenization treats each character in a word as a separate token. It has its uses, particularly in languages with simple morphologies or when fine-grained character-level analysis is necessary. However, it is less common in modern NLP models compared to other subword tokenization methods that create subword units for better handling of complex languages and word structures.

### SentencePiece

SentencePiece is a subword tokenization algorithm that's widely used in natural language processing (NLP) and text processing tasks. It's designed to segment text into subword units, making it useful for various languages, especially those with complex word structures and character-based writing systems. Here's how the SentencePiece algorithm works:

1. **Initialization**: Start with a basic vocabulary that typically includes individual characters, special symbols, and a few whole words. The vocabulary can be quite small initially.

2. **Training**: SentencePiece doesn't rely on specific linguistic knowledge or pre-existing word lists. Instead, it uses a statistical approach to analyze a training corpus of text data.

3. **Subword Segmentation**: SentencePiece applies an unsupervised subword segmentation algorithm to identify and group frequently co-occurring characters, character sequences, and subword units in the training data.

4. **Subword Units**: It creates a vocabulary of subword units, which can include character sequences, whole words, and more, based on their frequency in the training data. These subword units are used to tokenize text into meaningful pieces.

5. **Tokenization**: During tokenization for NLP tasks, the SentencePiece algorithm uses the vocabulary of subword units to segment text into these units. For example, the word "unbreakable" might be segmented into "un," "break," and "able."

6. **Flexibility**: SentencePiece allows for different tokenization strategies, including subword units that span characters or longer subword sequences. It can also handle languages with complex morphology and character-based writing systems effectively.

7. **Integration**: SentencePiece is often used in various NLP models, including BERT and other transformers, as it offers flexibility in handling subword tokenization and helps in representing words and subword units effectively.

SentencePiece is valuable in NLP for several reasons:

- It dynamically adapts to the specific language and data, making it suitable for languages with complex morphologies.
- It can represent out-of-vocabulary words and handle unseen or rare words effectively.
- It offers flexibility in choosing the granularity of subword units, making it adaptable to different NLP tasks and languages.

Overall, SentencePiece is a versatile and powerful subword tokenization algorithm that enhances the handling of text data in NLP tasks by segmenting it into meaningful subword units.


## Sampling

These are the techniques that are used to select a subset of the data for the task. The requirement is that the subset of the data should represent the original population. The selected sub dataset is called a sample. There are different ways of sampling which are mentioned below:

1. **Random sampling**: Select a subset in a pure random fashion
2. **Systematic sampling**: Choose the subset of data in a systematic manner at a regular interval
3. **Stratified sampling**: Population is divided into homogenous groups and then sampling is done from those groups to represent the original population
4. **Cluster sampling**: Data is divided into clusters and pure random sampling is done over clusters to select the subset
5. **Multistage sampling**: Apply different sampling techiques together
6. **Quota sampling**: Non probabilistic method where sub population is selcted based on their representation of the real population
7. **Convenience sampling**: Data collection from population members, non probablistic method
8. **Snowball sampling**: Existing popualation recruit future subjects probably their acquiantances

Sampling is a very important step as it will directly impact the model to which the sampled data will be fed.


## Token Sampling (Source: [Link](https://aman.ai/primers/ai/token-sampling/) )

Token sampling is one of the key techniques which the language models use to generate the next token. We already know that language model works just to generate the next probabilistically best token. There are several ways in which that can be done. Before that we will look at few key terminalogies:

### Key terms used

**Autoregressive Decoding:**
- Begin with a textual prefix/prompt.
- Generate the next token using the language model.
- Add the output token to the input sequence.
- Repeat this process to generate the entire textual sequence.

**Token Probabilities:**
- Language models output a probability distribution over all possible tokens.
- They function as neural networks solving a classification problem over the vocabulary.
- Strategies for selecting the next token include greedy decoding, choosing the token with the highest probability.

**Logits and Softmax:**
- Language models produce logits (z) and use the softmax function to generate a probability distribution (q).
- Softmax normalizes logits, ensuring values are between zero and one, making them interpretable as probabilities.

**Related: Temperature:**
- Temperature is a hyperparameter affecting token sampling.
- It adjusts the probability distribution over tokens.
- Higher temperature increases randomness and diversity in predictions but may lead to more mistakes.
- Lower temperature makes the model more conservative and focused.

**Temperature in Softmax:**
- Temperature (T) is a parameter in the softmax function.
- For T=1, softmax is directly computed on logits.
- A lower temperature (e.g., 0.6) results in a more confident but conservative model.
- Higher temperature produces a softer distribution, making the model more easily excited and diverse.

**Impact of Temperature:**
- Higher temperature increases sensitivity to low probability candidates.
- Output can be a letter, word, pixel, etc., depending on the task.
- High temperature leads to nearly equal probabilities, low temperature prioritizes the sample with the highest expected reward.

### Key token sampling techniques

**Greedy Decoding:**
- Selects the output with the highest probability at each decoding step.
- May produce suboptimal results as it lacks the ability to rectify earlier mistakes in the sequence.
- Efficient but not always the best choice due to its myopic decision-making.

**Exhaustive Search Decoding:**
- Considers every possible output sequence, choosing the one with the highest score.
- Computationally intensive and impractical for real-world applications.
- Time complexity of O(V(T)), where V is vocab size and T is sequence length.

**Beam Search:**
- Efficient algorithm exploring multiple possibilities.
- Maintains k most probable partial candidates (beam size) at each decoding step.
- Not guaranteed to find the optimal solution but more practical than exhaustive search.

**Constrained Beam Search:**
- Adds constraints to generated sequences to meet specific criteria.
- Modifies traditional beam search by incorporating constraints.
- Ensures valid sequences while maintaining diversity and fluency.
- Commonly used in tasks like text summarization, machine translation, and dialogue generation.

**Top-k:**
- Samples from a shortlist of top k tokens.
- Balances diversity and control in output.
- Higher k values result in increased diversity and less control.
- Suitable for text generation and conversational AI.

**Top-p (Nucleus Sampling):**
- Dynamically sets the size of the shortlist of tokens based on a threshold p.
- Chooses from the smallest set whose cumulative probability does not exceed p.
- Offers a balance between diversity and control.
- Controlled by setting the top-p parameter in language model APIs.

### Additional points

**Nucleus Sampling:**
- Useful for tasks requiring fine-grained control over diversity and fluency, like language modeling and text summarization.
- Top-p sets a threshold to limit the long tail of low probability tokens (often set around 75%).
- Dynamically adjusts the number of tokens considered during decoding based on their probabilities.
- Can be used simultaneously with top-k, with top-p applied after top-k.
- Provides flexibility in controlling the randomness of a language model's output.

**Nucleus Sampling vs. Temperature:**
- Nucleus sampling and temperature are distinct methods for controlling randomness in a language model's output.
- In the GPT-3 API, nucleus sampling and temperature cannot be used together.
- Temperature influences the softmax function, adjusting the overall randomness of predictions.
- Nucleus sampling, on the other hand, dynamically adjusts the number of tokens considered based on their probabilities.
- Choose between nucleus sampling and temperature based on the desired control over randomness in the model's output.
