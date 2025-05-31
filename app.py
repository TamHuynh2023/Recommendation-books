# from openai import OpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings



embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key = api_key)
vector_store = FAISS.load_local("D:/Programming/Mine/AI-DataScience/Book_recommendation/faiss_vector_store", embeddings,
                                allow_dangerous_deserialization=True)


import pandas as pd
books_with_emotions = pd.read_csv("D:/Programming/Mine/AI-DataScience/Book_recommendation/books_with_emotions/book_with_emotions.csv")

def retriever_book_from_database(query: str, tone: str = None, category = None, k: int = 20) -> pd.DataFrame:
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    isbn_list = []
    for doc in docs:
        try:
            isbn = int(doc.page_content.split()[0].strip())
            isbn_list.append(isbn)
        except (ValueError, IndexError):
            continue


    books_query = books_with_emotions[books_with_emotions["bookISBN"].isin(isbn_list)].head(k)
    if category != "All":
        books_query = books_query[books_query["bookGenres"].isin([category])].head(k)
    else:
        books_query = books_query.head(20)


    if tone == "Happy":
        books_query = books_query.sort_values(by="joy", ascending=False)
    elif tone == "Sad":
        books_query = books_query.sort_values(by="sadness", ascending=False)
    elif tone == "Angry":
        books_query = books_query.sort_values(by="anger", ascending=False)
    elif tone == "Fear":
        books_query = books_query.sort_values(by="fear", ascending=False)
    elif tone == "Neutral":
        books_query = books_query.sort_values(by="neutral", ascending=False)
    elif tone == "Surprise":
        books_query = books_query.sort_values(by="surprise", ascending=False)
    else:
        books_query = books_query.sort_values(by="disgust", ascending=False)

    return books_query




def query_books(query: str, category: str, tone: str):
    try:
        test = retriever_book_from_database(query, tone, category)
        authors = test["bookAuthors"].apply(
            lambda x: ", ".join(x.split()[:3]) + "....." if len(x.split()) > 3 else x
        )
        title = test["bookTitle"].apply(
            lambda x: ", ".join(x.split()[:8]) + "....." if len(x.split()) >= 8 else x
        )
        description_book = test["bookDesc"].apply(
            lambda x: ", ".join(x.split()[:30]) + "....." if len(x.split()) >= 30 else x
        )
        captions = (
            "Author: " + authors + "\nTitle: " + title + "\nDescription: " + description_book
        )
        information_books = list(zip(test["bookImage"], captions))
        return information_books
    except Exception as e:
        return [("", f"Error: {str(e)}")]


categores = ["All"] + sorted(books_with_emotions["bookGenres"].unique())
tones = ["All"] + ["Happy", "Fear", "Angry", "Surprise", "Sad", "Disgust", "Neutral"]



import gradio as gr
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Book Recommendation")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categores, label = "Select a category" , value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit = gr.Button("Find recommendations")

    gr.Markdown("## Recommendation")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit.click(fn = query_books,
                 inputs = [user_query, category_dropdown, tone_dropdown],
                 outputs = output)


dashboard.launch()

