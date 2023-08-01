---
layout: post
title: "Using LLMs with retrieval augmentation through LangChain"
date: 2023-07-30 8:00 +0000
categories: Software LLM langchain Python
tags: Software LLM langchain Python
---

## Intro 
This is going to be the first post in a series of posts on using the [LangChain][7] framework.
With LangChain you can develop apps that are powered by Large Language Models (LLMs).
I primarily use LangChain to build applications for chatting about literature in my Zotero library and other text data on my computer.
My intention was to write a blog post that explains how to build these applications and how they work.
However, there is too much ground to cover for a single post, so I decided to break it down into multiple posts.

This first post covers:
1. Background on how the development of LLM-powered applications started for me.
2. Application of LLM-powered that interests me most right now: [retrieval augmentation][14]
3. Explanation of how I implement some of the steps in retrieval augmentation, like creating and maintaining [vector stores][18] for ingesting documents that we want to chat about. 
In future posts I will discuss other parts of the process, such as setting up chatbots using LangChain for discussing documents and creating [agents][12] to take things further.

## Mind-blown
When ChatGPT was first released, I barely took notice.
I heard some people say impressive things about ChatGPT, but I didn't immediately feel the urge to try it out.
Eventually, a few months ago, I decided to give it a try.
I was mind-blown.
For an entire week, I had ChatGPT spit out crazy, nonsensical stories.
There was one about a talking horse that specialized in public-private partnerships and saved a village by helping to create new infrastructure.
I remember one about a hero who rode his horse backwards because he was afraid of being followed by purple frogs.
There was also another one about someone who put themselves in orbit around the Earth by pulling themselves up by their own hair.
The funniest part was that ChatGPT added a disclaimer by the end, stating that it was purely fictional and that you cannot actually put yourself in orbit in this way.

I quickly started trying out things that might be useful for my work in academia.
For example, I had ChatGPT come up with an assignment about analyzing policy interventions from a behavioral perspective.
Although I didn't actually use it, I could have with just a few tweaks.
I also entered into discussions with ChatGPT about theories and philosophy.
I found ChatGPT to be a useful conversational partner on many topics, as long as you already know your stuff and can spot the things that ChatGPT gets wrong (which happens frequently; at some point, I got fed up with ChatGPT constantly apologizing for getting things wrong).
I even tried a hybrid of storytelling and conversation on theories, having ChatGPT tell a fictional story about an academic and then querying ChatGPT about the contents of the papers written by this fictional academic.

I don't remember exactly when I started using ChatGPT for code writing, but its co-pilot capabilities are another aspect that blew me away and changed the way I write code. 
I recently read [a post on hacker news][1] about how traffic on StackOverflow has declined recently. 
I have a strong suspicion that ChatGPT has contributed to this. 

While I continue to be mind-blown to this day (in a positive way), I would also like to note that, like many others, I have occasionally felt uncertain and worried about what this will all lead to.
I am certainly no expert on AI, so please take anything I say on this with a grain of salt.
That being said, I am not that concerned about the 'AI going rogue' scenario, because I think that tools like ChatGPT give the strong appearance of being intelligent, but in reality are as dumb as a bag of rocks. 
What I am more afraid of [what humans might do][2] with powerful tools like LLMs (or whatever comes next).
Also, I feel somewhat uncomfortable with the fact that progress in this area is driven almost entirely by business interests.
I think it is important that we think of alternative models for the further development of AI, such as the 'Island' idea put forward in [this article of the FT Magazine][3]
It is also encouraging to see initiatives such as the development of [the Bloomz model][4] and [petals][5] (my GPU is now hosting one block of a BloomZ model through petals; admittely an almost meaningless contribution), which are both initiatives of [BigScience][6]. 
Yet, in my limited experience OpenAI's GPT models blow models such as Bloomz out of the water when it comes to the quality of their output.
The debate on how this AI revolution should unfold and be governed is an important one, but it is not a debate I want to engage with in these posts.
I would like to focus on different ways in which we can make LLMs useful for academic work.

<blockquote>
I'll reiterate that I am no expert on AI, and given that many people who are an expert on the topic are worried, probably means that I am overlooking or misunderstanding something.
Feel free to write me a message to educate me on this.
</blockquote>

## Down the LLM rabbit hole with LangChain
As described above, my first introduction to LLMs was through ChatGPT, which I believe is the case for many others.
While I had a lot of fun with ChatGPT alone, things became even more interesting after I discovered [LangChain][7].
I was introduced to LangChain by a [Youtube video][8].
In the video, Techlead demonstrates how LangChain allows you to chat with Language Models (LLMs) about data stored on your own computer.
Techlead also provides a [simple example][9] on his GitHub repository, which can help you get started even if you don't fully understand how LangChain works.
As mentioned in the introduction, you can use LangChain to develop LLM-powered apps.
These LLMs can run on your own computer or be accessed via APIs.
The apps I have created using LangChain so far make use of the OpenAI API, which provides access to chat models like `gpt3.5-turbo` and `gpt4`, as well as the `text-embedding-ada-002` embedding model (Using the OpenAI API is [not for free][10]).

As the name suggests, LangChain utilizes chains, which [the documentation][11] defines as sequences of calls to components. These components are abstractions for working with language models, along with various implementations for each abstraction. 
In simple terms, LangChain a set of tools that allow you to interact with LLMs in different ways, and it offers an easy way to chain these tools together.
This enables you to accomplish complex tasks with minimal code.
The LangChain comes with a wide variety of pre-built chains, which means that you can build useful tools quickly.

LangChain also allows the creation of [agents][12], which are LLMs that can choose actions based on user prompts.
Agents simulate reasoning processes to determine which actions to take and in what order.
These actions often involve using [tools][13] that we provide to the agent, which are different types of chains powered by LLMs.
In simple terms, an agent is an LLM that can use other types of LLMs for different tasks, depending on the specific needs.
There are undoubtedly many different kinds of useful applications that you can build with this framework, but I was drawn primarily to the idea of 'chatting with your own data'.
This involves something that is called [retrieval augmentation][14].

## Retrieval augmentation
With retrieval augmentation, you extract information from documents and include that information as context in the messages that you send to an LLM. 
This allows the LLM to not only make use of the knowledge that it obtained during training ([parametric knowledge][15]), but also of 'external' knowledge that you extract from the documents (source knowledge).
Supposedly, this helps to combat so-called [hallucinations][16] (or [check this link][17] if you don't have a Medium account).
That in itself is useful, but I was primarily enthusiastic about the idea of chatting with an LLM about the literature that I have collected in my Zotero library.

While the idea of extracting information from documents to include them as context in your messages to LLMs is simple enough, there are some challenges we need to overcome:

First, it is not practical if we have to manually find and extract the relevant information from our documents.
We might not even know exactly which information from which documents is relevant to our query in the first place.
Obviously, this part of the process is something we want to automate, which fortunately is easy using retrieval augmentation.

Second, there are limits to how much context we can include in our messages to LLMs.
Every LLM model has something called a context window, which refers to the number of tokens we can use in a single interaction with an LLM, including both the input (our query) and the output (the LLM's answer) of that interaction.
Different models have differently sized context windows.
For example, the `gpt3.5-turbo` model has a context window of 4,096 tokens.
The slightly more expensive `gpt3.5-turbo-16k` model, which I now use as my default, has a context window of 16,384 tokens.
The `gpt-4-32k` model has a context window of 32,768 tokens, but it is much more expensive than the `gpt3.5` models.
Anthropic's Claude 2, only available in the US and the UK, has an impressive context window size of 100k tokens!
Regardless, the length of the text that you include in your messages as context is limited by the model's context window.
If we want to ask questions about our literature, we cannot simply dump our entire library of papers into our messages.

Third, we might not want to dump our entire library, or even an entire book or paper, in our messages for another reason: 
Not all information in a given paper will be relevant to the question we are asking to the LLM. 
It would be preferable to just include the relevant bits of information in our messages and leave out everything that possibly only distracts from our question.
This too can fortunately be easily achieved, with tools provided by the LangChain framework.
I will now discuss some of these tools.

## Vector stores
[Vector stores][18] are perhaps the most important tools in the process of retrieval augmentation. 
A vector store is a kind of database in which you can store documents in two forms:
1. The actual documents in textual form, along with metadata.
2. The documents in their 'embedded' form, which is numerical representation of the documents.
In their embedded form, documents are stored as vectors that represent their position in a high-dimensional semantic space (the closer texts are in this space, the more similar they are in their meaning).
For example, OpenAI's `text-embedding-ada-002` model turns documents into vectors with 1,536 dimensions.

LangChain supports a variety of vector stores.
The one I chose to use is the FAISS vector store, for the following reasons:
1. It allows you to keep your vector stores on your local drive (it doesn't require a cloud solution).
2. For my purposes it is important that I can easily save, load and update a vector store and I found the approach that the FAISS vector store takes to this to be the most intuitive.

Another type of vector store that offers similar functionality is ChromaDB, which also seems to be popular.
I advise you to [explore][19] the different available types of vector stores in LangChain before picking one to use yourself.

To store our documents in a vector store we need to take multiple steps (I'll walk through these in more detail in the remainder of this blog post):
1. We need to convert our documents to plain text (assuming that many of them will be in PDF-format originally).
LangChain includes PyPDF-based tools that will do this for you, but I opted to convert my files using a bash script that utilizes `pdfttotext` and `pdfimage`. 
This is another thing that you might want to do different (you might want to simply make use of the built-in tools that LangChain provides for this). 
I opted for the bash script because it makes it easier to check the results of the conversion process.
2. We need to load our documents into our application, for which LangChain again offers multiple solutions.
We'll use the `DirectoryLoader` (see further below) as we'll be loading multiple documents from a directory.
3. We might want to add metadata to our documents, which can be stored along with the documents in our vector store. 
For example, I like to add the bibliographical details of the publications in my Zotero library.
We will want to do this before we cut up our documents in smaller chunks (the next step).
4. We will want to cut our documents into smaller chunks that we than store separately in our vector store.
When we retrieve information from our vector store, we'll thus retrieve these smaller chunks, rather than the entire original documents.
This allows us to retrieve relevant information in a more targeted way, as well as limit the amount of text we include as context in our messages to LLMs (see our earlier discussion on challenges in extracting contextual information on documents).
LangChain comes with multiple text splitters that can accomplish this task for us.
We'll be using the `RecursiveCharacterTextSplitter`.
5. We need to create the embeddings for our chunks of texts, for which we will use OpenAI's `text-embedding-ada-002`.
Again, there are other options available, but I haven't had the chance to experiment with these yet and I'm quite happy with the results I have achieved with the OpenAI solution.
6. Then we have everything we need to store our documents, along with their embeddings, in our vector store.

Let's now go through these steps in more detail.

## Converting to text
I'll briefly explain the logic of the bash script (which you can find below) that I use to convert the literature in my Zotero library (all PDFs) to text.

The bash script finds all the PDFs that are include in my Zotero storage folder, which has sub-folders for each publication.
For each file it checks if the filename is already mentioned in a text file that I use to keep track of every document that I have already ingested.
I frequently update my Zotero library, and if I want update my vector store by adding new publications, I don't want to also convert all files that have already been ingested.

``` bash
#!/bin/bash

# One file to keep the papers that I have already ingested
# One dir to store newly added papers
# A temporary dir for image-based pdfs.
existing_file="/home/wouter/Documents/LangChain_Projects/Literature/data/ingested.txt"
output_dir="/home/wouter/Documents/LangChain_Projects/Literature/data/new"
temp_dir="/home/wouter/Documents/LangChain_Projects/Literature/data/temp"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)

find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if grep -Fxq "$base_name.txt" "$existing_file"; then
	echo "Text file for $file already exists, skipping."
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"

	pdfimages "$file" "$temp_dir/$base_name"
	
    fi
    counter=$((counter + 1))
    echo -ne "Processed $counter out of $total PDFs.\r"
    
done
```
I have the bash script convert all PDFs to text with `pdftotext`, but I also convert the same files with `pdfimages`, since some of the PDFs have images rather than text (the PDFs where you cannot select the text).
The images are stored to a temporary folder.
After converting the files, I basically just inspect the resulting files and try to identify files that `pdftotext` was not able to convert successfully (usually these are just a few bytes in size).

For all files that *were* converted successfully I delete the image files in the temporary folder.
The remaining image files are converted with another bash script, which makes use of tesseract.

<blockquote>
One really cool benefit of storing documents in their vectorized form is that the language in which the documents were written becomes less relevant. 
Two documents that are written in different languages, but have similar meanings, will end up in similar positions in the semantic space when they are embedded.
</blockquote>


``` bash
#!/bin/bash

output_dir="/home/wouter/Documents/LangChain_Projects/Literature/data/new/"
pbm_directory="/home/wouter/Documents/LangChain_Projects/Literature/data/temp"

# Create an associative array
declare -A base_names

# Handle filenames with spaces by changing the Internal Field Separator (IFS)
oldIFS="$IFS"
IFS=$'\n'

# Go through each file in the PBM directory
for file in "$pbm_directory"/*.pbm "$pbm_directory"/*.ppm
do
    # Get the base name from the path
    base_name=$(basename "$file" | rev | cut -d- -f2- | rev)

    # Add the base name to the associative array
    base_names["$base_name"]=1
done

# Restore the original IFS
IFS="$oldIFS"

# Go through each unique base name
for base_name in "${!base_names[@]}"
do
    # Remove any existing text file for this base name
    rm -f "$output_dir/$base_name.txt"

    # Go through each PBM file for this base name, handling spaces in filenames
    for ext in pbm ppm
    do
        find "$pbm_directory" -name "$base_name-*.$ext" -print0 | while read -r -d $'\0' file
        do
            # OCR the file and append the results to the text file
	    echo "Converting $file" 
            tesseract "$file" stdout >> "$output_dir/$base_name.txt"
        done
    done
done
```
After doing this, I have all documents stored in plain text in one folder.

## Continuing in Python
The remaining steps that I discuss in this post are all implemented in Python.
In the remainder of this post I share the Python script that I call `indexer.py`, which I use to create a new vector store for the literature in my Zotero library.
I'll be going through the script in steps.
Let's start with some basic 'housekeeping' stuff, such as imports, loading our OpenAI API Key (without it, we cannot use the OpenAI models), and setting some paths that we'll be using throughout the script. 

``` python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import langchain
import bibtexparser
import os
import glob
from dotenv import load_dotenv
import openai
import constants
import time

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Set paths
source_path = './data/new/'
store_path = './vectorstore/'
destination_file = './data/ingested.txt'
bibtex_file_path = '/home/wouter/Tools/Zotero/bibtex/library.bib'
```
You'll notice that I import my API key from a file called `constants.py`, which is a file that just defines one variable, called `APIKEY`, which is a string that contains my API key. 
If you don't have an OpenAI API key yet, you can make one on the [OpenAI platform][20].
It is important that you don't share your API key with anyone.

In the snippet of Python code above we set a couple of paths:
- The `source_path` which contains all the text files we created in the previous step.
- The `store_path` where we will keep our vector store.
- The `destination_file` to which we'll write the names of all the files we've successfully ingested later on.
- The `bibtex_file_path` where we store our Zotero-generated bibtex file.
We will access this file to retrieve the bibliographical metadata that we want to include with our documents.

## Loading our documents and adding metadata
The next step is to actually load our documents, which we can easily accomplish with LangChain's [DirectoryLoader][21].
Before chunking our documents we will also want to add the metadata to them, so that the metadata is associated with the relevant chunks.

We simply setup our `DirectoryLoader`, passing our `source_path` as its first argument and then setting a few options that help ensure a smooth process (the `show_progress=True` argument is not strictly necessary).

To add our metadata, we can go through our bibtex file, using the `bibtexparser` library, and we'll match the names of our documents to the filenames recorded in the bibtex file (Zotero conveniently records these names along with the bibliographical details).
After extracting the metadata, we go through our list of the imported documents, and we add the metadata to the matching documents.

``` python
# Load documents
print("===Loading documents===")
loader = DirectoryLoader(source_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs={'autodetect_encoding': True})
documents = loader.load()

# Add metadata based in bibliographic information
print("===Adding metadata===")

# Read the BibTeX file
with open(bibtex_file_path) as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Get a list of all text file names in the directory
text_file_names = os.listdir(source_path)
metadata_store = []

# Go through each entry in the BibTeX file
for entry in bib_database.entries:
    # Check if the 'file' key exists in the entry
    if 'file' in entry:
        # Extract the file name from the 'file' field and remove the extension
        pdf_file_name = os.path.basename(entry['file']).replace('.pdf', '')

         # Check if there is a text file with the same name
        if f'{pdf_file_name}.txt' in text_file_names:
            # If a match is found, append the metadata to the list
            metadata_store.append(entry)

for document in documents:
    for entry in metadata_store:
        doc_name = os.path.basename(document.metadata['source']).replace('.txt', '')
        ent_name = os.path.basename(entry['file']).replace('.pdf', '')
        if doc_name == ent_name:
            document.metadata.update(entry)
```

## Splitting the documents

Now that we have our documents, including metadata, we can start splitting them.
As mentioned previously, we can use the [RecursiveCharacterTextSplitter][21] for this, which is very good at splitting texts into chunks of the size that we desired, while keeping semantically meaningful structures (e.g., paragraphs) intact as much as possible.

We need to decide what the size of our chunks will be.
I believe a popular choice is to go with chunks of 1000 tokens.
I opted for 1500 tokens because it just slightly increases the chances that parts of the text that belong together also end up in chunks together.
We can also set an overlap for our chunks, which I set to 150.

``` python
# Splitting text
print("===Splitting documents into chunks===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
    length_function = len,
    add_start_index = True,
)

split_documents = text_splitter.split_documents(documents)
```

## Embedding the documents and creating our vector store
The final steps are to create embeddings for our chunks of texts and then store them, alongside the chunks themselves, in our vector store.
These embeddings are what we actually use later when we want to retrieve information from our vector store (discussed in more detail in a future post).
Basically, when we ask our LLM a question, the question will be embedded as well and its vectorized form will then be used to find entries in our vector store that are similar in meaning.
This approach to identifying relevant information is much more accurate than finding relevant information purely based on matches between the texts themselves.

<blockquote>
One thing about embeddings that I think is very cool, is that texts that are written in different languages, but that have similar meanings, will also be similar in their embedded form.
</blockquote>

As mentioned previously, we use the `text-embedding-ada-002` model to create our embeddings. 
This is the default model when using LangChain's `OpenAIEmbeddings()` function.

Creating the embeddings is the most time consuming part of this process. 
I started out with a library of about 1750 documents (before chunking), which I believe takes about an hour to complete the embeddings for (this is a guess, because I didn't consciously keep track of time).
It is also a relatively expensive part of the process, since we'll be sending a lot of tokens through the OpenAI API.
This is also one of the reason's why it is useful to have a setup where you don't have to recreate these embeddings over and over (see further below).

You will probably also frequently see warnings about hitting OpenAI's rate limits. 
Fortunately, LangChain has built-in functions that delay further requests until we're ready to resume the process.

After the embeddings have been created, you can create your vector store as shown in the snippet. 
We immediately save our vector store in the path that we defined for it earlier.

The last thing that we do is to write the filenames of the ingested documents to the file that we use to keep track of all ingested documents.
``` python
# Embedding documents
print("===Embedding text and creating database===")
embeddings = OpenAIEmbeddings(
    show_progress_bar=True,
    request_timeout=60,
)

db = FAISS.from_documents(split_documents, embeddings)
db.save_local(store_path, "index")

# Record what we have ingested
print("===Recording ingested files===")
with open(destination_file, 'w') as f:
    for document in documents:
        f.write(os.path.basename(document.metadata['source']))
        f.write('\n')
```            

## Updating the vector store
As mentioned above, creating embeddings for documents is relatively expensive, both in terms of time and in terms of actual money spent on using the OpenAI API.
Therefore, we do not want to create embeddings for any given document more than once.
I already explained how the bash script that I use to convert PDFs skips documents that we've already ingested.
If I add new papers to my Zotero library, and I run the conversion script, only the PDFs of the newly added papers will end up in the relevant folder of text files.

To add these new papers to my existing vector store, I use a script that I named `updater.py` (see below).
This script is identical to the `indexer.py` script, except for the last part, where I:
1. create a new vector store to ingest the new papers,
2. load the existing vector store that I initially created with the `indexer.py` script,
3. merge these two vector stores, and
4. store the merged vector store to my disk, overwriting the original one.

This process requires me to only create the embeddings for any new papers that I add to my library.
``` python
print("===Embedding text and creating database===")
new_db = FAISS.from_documents(split_documents, embeddings)

print("===Merging new and old database===")
old_db = FAISS.load_local(store_path, embeddings)
old_db.merge_from(new_db)
old_db.save_local(store_path, "index")

# Record the files that we have added
print("===Recording ingested files===")
with open(destination_file, 'a') as f:
    for document in documents:
        f.write(os.path.basename(document.metadata['source']))
        f.write('\n')
```
## Outlook to future posts
This is all that I wanted to share in this particular post.
What we have done now is to create a vector store that includes (for example) literature in our Zotero library, which allows us to then use that literature as context in chat sessions with LLMs.
How we actually set up these chat sessions and how we can use the vector stores in them is something I will discuss in a future post.

[1]: https://news.ycombinator.com/item?id=36855516
[2]: https://www.economist.com/by-invitation/2023/07/21/one-of-the-godfathers-of-ai-airs-his-concerns
[3]: https://www.ft.com/content/03895dc4-a3b7-481e-95cc-336a524f2ac2
[4]: https://huggingface.co/bigscience/bloomz
[5]: https://github.com/bigscience-workshop/petals#benchmarks
[6]: https://bigscience.huggingface.co/
[7]: https://python.langchain.com/docs/get_started/introduction.html
[8]: https://www.youtube.com/watch?v=9AXP7tCI9PI
[9]: https://github.com/techleadhd/chatgpt-retrieval
[10]: https://openai.com/pricing
[11]: https://python.langchain.com/docs/modules/chains/
[12]: https://python.langchain.com/docs/modules/agents/
[13]: https://python.langchain.com/docs/modules/agents/tools/
[14]: https://www.promptingguide.ai/techniques/rag
[15]: https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/
[16]: https://towardsdatascience.com/llm-hallucinations-ec831dcd7786
[17]: https://machinelearningmastery.com/a-gentle-introduction-to-hallucinations-in-large-language-models/
[18]: https://python.langchain.com/docs/modules/data_connection/vectorstores/
[19]: https://js.langchain.com/docs/modules/data_connection/vectorstores/
[20]: https://platform.openai.com/
[21]: https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
[22]: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
