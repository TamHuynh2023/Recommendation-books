import pandas as pd

final_books = pd.read_csv("D:/Programming/Mine/AI-DataScience/Book_recommendation/utils/second_clean_csv/second_cleaned.csv")

final_books["year_published"] = final_books["year_published"].str.extract(r'(\d+)')
final_books.dropna(subset=['year_published'], inplace=True)
final_books['year_published'] = final_books['year_published'].astype(int)

final_books.drop_duplicates("bookTitle", inplace=True)

for value in final_books["isbn_description"]:
    with open("D:\Programming\Mine\AI-DataScience\Book_recommendation\database_for_query.txt", "a", encoding = "utf-8") as file:
        file.write(f"{value}\n")


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss


raw_documents = TextLoader("D:/Programming/Mine/AI-DataScience/Book_recommendation/database_for_query.txt", encoding = "utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)




embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key = api_key)

dimension = 3072  
index = faiss.IndexFlatL2(dimension)  
vector_store = FAISS(
    embedding_function=embeddings,  
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


texts = [doc.page_content for doc in documents]  
metadatas = [doc.metadata for doc in documents]  
vector_store.add_texts(texts=texts, metadatas=metadatas)
vector_store.save_local("D:/Programming/Mine/AI-DataScience/Book_recommendation/faiss_vector_store")


final_books.reset_index(drop=True, inplace=True)


# make emotions
from transformers import pipeline, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
pipe = pipeline("text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k = None,
                tokenizer=tokenizer,
                device=-1)

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
emotion_scores = {label: [] for label in emotion_labels}

for idx, i in enumerate(final_books["bookDesc"]):
    emotion_scores_per = {label: [] for label in emotion_labels}
    sentences = [s.strip() for s in i.split(".") if s.strip()]
    print(idx)
    for text in sentences:
        inputs = tokenizer(text, truncation=True, max_length=512,
                           padding=True,
                           return_tensors="pt")
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens = True)

        num_tokens = len(inputs["input_ids"][0])
        if num_tokens >= 512:
            continue

        sorted_emotion = sorted(pipe(input_text)[0], key=lambda x: x["score"], reverse=True)

        for j in sorted_emotion:
            emotion_scores_per[j["label"]].append(j["score"])


    for key, value in emotion_scores_per.items():
        emotion_scores[key].append(max(value))

# create df
emotion_df = pd.DataFrame(emotion_scores)
books_with_emotions = pd.concat([final_books, emotion_df], ignore_index=False, axis=1)
books_with_emotions.to_csv("D:\Programming\Mine\AI-DataScience\Book_recommendation\books_with_emotions\book_with_emotions.csv", index = False)
